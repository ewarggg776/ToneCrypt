import numpy as np
import soundfile as sf
import logging
import zlib
from reedsolo import RSCodec, ReedSolomonError

logger = logging.getLogger(__name__)

class ToneDecoder:
    def __init__(self, sample_rate=44100, segment_length=1024, tone_length=800, ecc_symbols=32):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.tone_length = tone_length
        self.rsc = RSCodec(ecc_symbols)
        
        self.freqs = np.linspace(1000, 5000, 8)
        self.fft_freqs = np.fft.rfftfreq(tone_length, 1/sample_rate)
        self.f_indices = [np.argmin(np.abs(self.fft_freqs - f)) for f in self.freqs]
        self.window = np.hanning(tone_length)
        
        # Precompute Preamble for correlation
        sync_t = np.arange(2048) / sample_rate
        self.preamble = 0.5 * np.sin(2 * np.pi * 1200 * sync_t) + 0.5 * np.sin(2 * np.pi * 2400 * sync_t)
        self.preamble *= np.hanning(2048)

    def _deinterleave(self, bits):
        """Reverses the scattering of bits."""
        rows = 32
        cols = len(bits) // rows
        return bits.reshape(cols, rows).T.flatten()

    def decode_wav(self, input_wav_path, output_path):
        """v3 Modem Decoder: Sync -> Demod -> Deinterleave -> ECC -> CRC32."""
        audio_data, _ = sf.read(input_wav_path)
        
        # 1. Synchronization (Cross-Correlation)
        # Find exactly where the preamble starts in the audio
        correlation = np.correlate(audio_data, self.preamble, mode='valid')
        start_idx = np.argmax(np.abs(correlation))
        
        # The body starts after preamble + 0.5s silence
        body_start = start_idx + 2048 + (self.sample_rate // 2)
        audio_body = audio_data[body_start:]
        
        # 2. OMT Demodulation
        num_segments = len(audio_body) // self.segment_length
        bits = np.zeros(num_segments * 8, dtype=np.uint8)

        for i in range(num_segments):
            start = i * self.segment_length
            segment = audio_body[start : start + self.tone_length]
            if len(segment) < self.tone_length: break
            
            fft_res = np.abs(np.fft.rfft(segment * self.window))
            noise_floor = np.median(fft_res)
            
            for bit_idx, f_idx in enumerate(self.f_indices):
                # Adaptive Threshold Detection
                if fft_res[f_idx] > (noise_floor * 5):
                    bits[i*8 + bit_idx] = 1

        # 3. De-interleaving
        deinterleaved_bits = self._deinterleave(bits)
        byte_data = np.packbits(deinterleaved_bits, bitorder='little').tobytes()

        # 4. RS Correction & Integrity Check
        try:
            # Decode ECC
            full_payload = self.rsc.decode(byte_data)[0]
            
            # Read Header
            payload_size = int.from_bytes(full_payload[:8], 'big')
            data_with_crc = full_payload[8 : 8 + payload_size]
            
            # Read CRC32
            stored_crc = int.from_bytes(data_with_crc[:4], 'big')
            actual_data = data_with_crc[4:]
            
            # Verify Integrity
            if zlib.crc32(actual_data) != stored_crc:
                logger.error("CRC32 Mismatch! Data is corrupted.")
            else:
                logger.info("CRC32 Verified. Restoration Perfect.")
                
            with open(output_path, 'wb') as f:
                f.write(actual_data)
                
        except (ReedSolomonError, Exception) as e:
            logger.error(f"v3 Decoding Failed: {e}")
            with open(output_path, 'wb') as f:
                f.write(byte_data)
