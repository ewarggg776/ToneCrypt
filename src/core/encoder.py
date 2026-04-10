import numpy as np
import soundfile as sf
import logging
import os
import zlib
from reedsolo import RSCodec

logger = logging.getLogger(__name__)

class ToneEncoder:
    def __init__(self, sample_rate=44100, segment_length=1024, tone_length=800, ecc_symbols=32):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.tone_length = tone_length
        self.rsc = RSCodec(ecc_symbols)
        
        self.freqs = np.linspace(1000, 5000, 8) 
        self.t_vec = np.arange(tone_length) / sample_rate
        
        window = np.hanning(tone_length)
        self.subcarriers = [0.12 * np.sin(2 * np.pi * f * self.t_vec) * window for f in self.freqs]
        
        sync_t = np.arange(2048) / sample_rate
        self.preamble = 0.5 * np.sin(2 * np.pi * 1200 * sync_t) + 0.5 * np.sin(2 * np.pi * 2400 * sync_t)
        self.preamble *= np.hanning(2048)

    def _interleave(self, bits):
        rows = 32
        cols = len(bits) // rows
        if len(bits) % rows != 0:
            padding = rows - (len(bits) % rows)
            bits = np.concatenate([bits, np.zeros(padding, dtype=np.uint8)])
            cols += 1
        return bits.reshape(rows, cols).T.flatten()

    def encode_file(self, input_path, output_wav_path, carrier_music_path=None, progress_callback=None):
        """v3 Modem Encoder with optional Steganographic Mixing."""
        with open(input_path, 'rb') as f: 
            raw_data = f.read()

        crc = zlib.crc32(raw_data)
        data_with_crc = crc.to_bytes(4, 'big') + raw_data
        header = len(data_with_crc).to_bytes(8, 'big')
        full_payload = header + data_with_crc
        ecc_data = self.rsc.encode(full_payload)
        
        bits = np.unpackbits(np.frombuffer(ecc_data, dtype=np.uint8), bitorder='little')
        interleaved_bits = self._interleave(bits)
        
        num_segments = len(interleaved_bits) // 8
        audio_body = np.zeros(num_segments * self.segment_length)

        for i in range(num_segments):
            segment_bits = interleaved_bits[i*8 : (i+1)*8]
            chord = np.zeros(self.tone_length)
            for bit_idx, bit_val in enumerate(segment_bits):
                if bit_val: chord += self.subcarriers[bit_idx]
            
            start = i * self.segment_length
            audio_body[start : start + self.tone_length] = chord
            if i % 500 == 0 and progress_callback: progress_callback(i / num_segments)

        silence = np.zeros(self.sample_rate // 2)
        data_audio = np.concatenate([silence, self.preamble, silence, audio_body, silence])

        # --- STEGANOGRAPHY MIXING ---
        if carrier_music_path and os.path.exists(carrier_music_path):
            carrier, sr = sf.read(carrier_music_path)
            # Convert to mono if stereo
            if len(carrier.shape) > 1: carrier = np.mean(carrier, axis=1)
            
            # Match lengths (loop or pad carrier)
            if len(carrier) < len(data_audio):
                repeats = (len(data_audio) // len(carrier)) + 1
                carrier = np.tile(carrier, repeats)[:len(data_audio)]
            else:
                carrier = carrier[:len(data_audio)]
            
            # Mix: Data is added as a low-level background "hiss/reverb"
            # 0.1 gain for data ensures it's audible enough for decoder but hidden in music
            final_audio = (carrier * 0.7) + (data_audio * 0.15)
        else:
            final_audio = data_audio

        sf.write(output_wav_path, final_audio, self.sample_rate)
        logger.info(f"v3 Encoding Complete: {output_wav_path}")
