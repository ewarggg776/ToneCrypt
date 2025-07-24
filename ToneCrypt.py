#!/usr/bin/env python3
"""
ToneCrypt: Audio-based folder encryption tool
Encrypts folders into WAV files using pitch shifts, noise, and reverb, with tones encoding decryption keys.
Supports decryption from files, microphone, or CD, with error correction and folder restoration.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import librosa
import shutil
import argparse
import zlib
from reedsolo import RSCodec
from colorama import init, Fore, Style
import sys
import hashlib
from pydub import AudioSegment
import subprocess
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize Reed-Solomon codec (10 bytes error correction)
rs = RSCodec(10)

# Initialize colorama for colored output
init()

# Package version
VERSION = "1.3.1"

def folder_to_binary(folder_path):
    """Convert a folder to binary data with metadata."""
    logger.info("Converting folder to binary...")
    try:
        metadata = []
        binary_data = ""
        base_path = os.path.abspath(folder_path)
        
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, base_path)
                size = os.path.getsize(file_path)
                metadata.append(f"{rel_path}:{size}")
        
        metadata_str = "|".join(metadata)
        metadata_bytes = metadata_str.encode('utf-8')
        metadata_binary = ''.join(format(byte, '08b') for byte in metadata_bytes)
        binary_data += metadata_binary + "11111111"
        
        total_files = sum(len(files) for _, _, files in os.walk(folder_path))
        file_count = 0
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as f:
                    byte_data = f.read()
                    binary_data += ''.join(format(byte, '08b') for byte in byte_data)
                file_count += 1
                logger.info(f"Processed file {file_count}/{total_files}...")
        
        binary_data = zlib.compress(binary_data.encode('utf-8'))
        binary_data = ''.join(format(byte, '08b') for byte in binary_data)
        
        binary_bytes = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
        rs_encoded = rs.encode(binary_bytes)
        binary_data = ''.join(format(byte, '08b') for byte in rs_encoded)
        
        return binary_data, metadata
    except Exception as e:
        logger.error(f"Error converting folder to binary: {e}")
        return None, None

def encode_pitch_shift(pitch_shift_hz, num_tones=3):
    """Encode pitch shift as multiple tones."""
    base_freq = 100
    tones = []
    target = pitch_shift_hz
    for i in range(num_tones):
        freq = base_freq + (i * 50) + (target / num_tones / 2)
        tones.append(min(freq, 2000))
        target -= freq / num_tones
    if num_tones >= 3:
        tones[0] = min(tones[0], 1000)
        tones[2] = max(tones[2], 100)
    return tones

def decode_pitch_shift(tones):
    """Decode pitch shift from tones."""
    if len(tones) < 2:
        return 0
    f1 = tones[0]
    f2 = tones[1]
    if len(tones) == 3:
        f3 = max(tones[2], 1)
        return np.cos(f1 / 200) * f2**3 + np.log(f3 + 1) * np.exp(f2 / 500) + np.sin(f1 / f3)
    return np.cos(f1 / 200) * f2**3

def binary_to_wav(binary_data, output_wav_path, pitch_shift_hz1, pitch_shift_hz2, num_tones=3, sample_rate=44100, noise_level=0.1):
    """Convert binary data to WAV with tones and noise."""
    logger.info(f"Encoding binary to WAV with {num_tones} tones...")
    try:
        duration_per_bit = 0.001
        freq_0 = 440
        freq_1 = 880
        tone_duration = 0.5
        
        t = np.arange(0, duration_per_bit, 1/sample_rate)
        t_tone = np.arange(0, tone_duration, 1/sample_rate)
        audio_signal = []
        
        tones1 = encode_pitch_shift(pitch_shift_hz1, num_tones)
        tones2 = encode_pitch_shift(pitch_shift_hz2, num_tones)
        for freq in tones1 + tones2:
            tone = np.sin(2 * np.pi * freq * t_tone)
            audio_signal.extend(tone)
        
        total_bits = len(binary_data)
        for i, bit in enumerate(binary_data[:total_bits//2]):
            freq = freq_0 if bit == '0' else freq_1
            signal = np.sin(2 * np.pi * freq * t)
            audio_signal.extend(signal)
            if i % 10000 == 0:
                logger.info(f"Encoded {i}/{total_bits//2} bits (segment 1)...")
        
        for i, bit in enumerate(binary_data[total_bits//2:]):
            freq = freq_0 if bit == '0' else freq_1
            signal = np.sin(2 * np.pi * freq * t)
            audio_signal.extend(signal)
            if i % 10000 == 0:
                logger.info(f"Encoded {i}/{total_bits//2} bits (segment 2)...")
        
        audio_signal = np.array(audio_signal)
        data_start = len(t_tone) * num_tones * 2
        data_signal = audio_signal[data_start:]
        data_signal = librosa.effects.pre_emphasis(data_signal, coef=0.95)
        audio_signal[data_start:] = data_signal
        
        noise = np.random.normal(0, noise_level * np.max(np.abs(audio_signal)), audio_signal.shape)
        freqs = np.fft.fftfreq(len(noise), 1/sample_rate)
        fft_noise = np.fft.fft(noise)
        fft_noise[(abs(freqs) < 400) | (abs(freqs) > 900)] *= 0.1
        noise = np.fft.ifft(fft_noise).real
        audio_signal = audio_signal + noise
        
        audio_signal = audio_signal * 32767 / max(abs(audio_signal))
        audio_signal = audio_signal.astype(np.int16)
        
        wavfile.write(output_wav_path, sample_rate, audio_signal)
        return output_wav_path, len(binary_data)//2, len(t_tone) * num_tones * 2
    except Exception as e:
        logger.error(f"Error encoding binary to WAV: {e}")
        return None, None, None

def convert_to_wav(input_path):
    """Convert audio file to WAV if necessary."""
    if input_path.lower().endswith('.wav'):
        return input_path
    try:
        temp_wav = "temp_input.wav"
        audio = AudioSegment.from_file(input_path)
        audio.export(temp_wav, format='wav')
        logger.info(f"Converted {input_path} to WAV")
        return temp_wav
    except Exception as e:
        logger.error(f"Error converting to WAV: {e}")
        return None

def record_from_mic(output_wav):
    """Record audio from microphone."""
    logger.info("Recording from microphone... Press Ctrl+C to stop.")
    try:
        subprocess.run(['arecord', '-D', 'plughw:0,0', '-f', 'cd', output_wav], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error recording from microphone: {e}")
        return None
    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")
    return output_wav

def rip_from_cd(output_wav, cd_drive='/dev/sr0'):
    """Rip audio from CD."""
    logger.info(f"Ripping audio from CD at {cd_drive}...")
    try:
        subprocess.run(['cdparanoia', '-B', f'-d {cd_drive}', '1', output_wav], check=True)
        logger.info(f"Ripped audio to {output_wav}")
        return output_wav
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ripping from CD: {e}")
        return None

def detect_cd_drive():
    """Detect CD drive."""
    try:
        result = subprocess.run(['lsblk', '-d', '-o', 'NAME'], capture_output=True, text=True)
        devices = result.stdout.splitlines()
        for device in devices:
            if device.startswith('sr'):
                return f"/dev/{device}"
        logger.warning("No CD drive detected.")
        return None
    except Exception as e:
        logger.error(f"Error detecting CD drive: {e}")
        return None

def encrypt_wav_with_pitch(input_wav_path, output_wav_path, pitch_shift_hz1, pitch_shift_hz2, segment_length, tone_length, num_tones=3, sample_rate=44100):
    """Encrypt WAV with pitch shifts."""
    logger.info(f"Encrypting WAV with pitch shifts ({pitch_shift_hz1:.0f} Hz, {pitch_shift_hz2:.0f} Hz)...")
    try:
        audio_data, sr = sf.read(input_wav_path)
        
        tones = audio_data[:tone_length]
        data = audio_data[tone_length:]
        
        segment1 = data[:segment_length]
        segment2 = data[segment_length:]
        
        segment1_shifted = librosa.effects.pitch_shift(segment1, sr=sr, n_steps=pitch_shift_hz1 / 100.0)
        segment2_shifted = librosa.effects.pitch_shift(segment2, sr=sr, n_steps=pitch_shift_hz2 / 100.0)
        
        encrypted_data = np.concatenate([segment1_shifted, segment2_shifted])
        
        if len(encrypted_data) > len(data):
            encrypted_data = encrypted_data[:len(data)]
        elif len(encrypted_data) < len(data):
            encrypted_data = np.pad(encrypted_data, (0, len(data) - len(encrypted_data)), 'constant')
        
        encrypted_audio = np.concatenate([tones, encrypted_data])
        encrypted_audio = (encrypted_audio * 32767 / max(abs(encrypted_audio))).astype(np.int16)
        
        wavfile.write(output_wav_path, sample_rate, encrypted_audio)
        return output_wav_path
    except Exception as e:
        logger.error(f"Error encrypting WAV: {e}")
        return None

def detect_tone_frequencies(audio, sr, tone_duration, num_tones):
    """Detect tone frequencies from audio."""
    tone_length = int(tone_duration * sr)
    tones = []
    for i in range(num_tones):
        start = i * tone_length
        end = (i + 1) * tone_length
        tone = audio[start:end]
        
        fft = np.fft.fft(tone)
        freqs = np.fft.fftfreq(len(tone), 1/sr)
        fft_magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(fft)//2]
        
        peaks, _ = signal.find_peaks(fft_magnitude, height=0.2*np.max(fft_magnitude), distance=10)
        if peaks.size > 0:
            peak_freq = abs(freqs[peaks[0]])
            tones.append(peak_freq)
        else:
            tones.append(0)
    return tones

def decrypt_wav_with_pitch(input_wav_path, output_wav_path, segment_length, tone_length, num_tones=3, sample_rate=44100):
    """Decrypt WAV with pitch shifts."""
    logger.info("Decrypting WAV with pitch shifts...")
    try:
        audio_data, sr = sf.read(input_wav_path)
        
        tone_duration = tone_length // (num_tones * 2)
        tones1 = detect_tone_frequencies(audio_data[:tone_length//2], sr, tone_duration, num_tones)
        tones2 = detect_tone_frequencies(audio_data[tone_length//2:tone_length], sr, tone_duration, num_tones)
        
        pitch_shift_hz1 = decode_pitch_shift(tones1)
        pitch_shift_hz2 = decode_pitch_shift(tones2)
        logger.info(f"Detected pitch shifts: {pitch_shift_hz1:.0f} Hz, {pitch_shift_hz2:.0f} Hz")
        
        data = audio_data[tone_length:]
        
        segment1 = data[:segment_length]
        segment2 = data[segment_length:]
        
        segment1_restored = librosa.effects.pitch_shift(segment1, sr=sr, n_steps=-pitch_shift_hz1 / 100.0)
        segment2_restored = librosa.effects.pitch_shift(segment2, sr=sr, n_steps=-pitch_shift_hz2 / 100.0)
        
        decrypted_data = np.concatenate([segment1_restored, segment2_restored])
        
        if len(decrypted_data) > len(data):
            decrypted_data = decrypted_data[:len(data)]
        elif len(decrypted_data) < len(data):
            decrypted_data = np.pad(decrypted_data, (0, len(data) - len(decrypted_data)), 'constant')
        
        decrypted_audio = decrypted_data.astype(np.int16)
        wavfile.write(output_wav_path, sample_rate, decrypted_audio)
        return output_wav_path
    except Exception as e:
        logger.error(f"Error decrypting WAV: {e}")
        return None

def binary_to_folder(binary_data, output_folder):
    """Restore folder from binary data."""
    logger.info("Restoring folder from binary...")
    try:
        binary_bytes = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
        binary_bytes = rs.decode(binary_bytes)[0]
        
        binary_data = zlib.decompress(binary_bytes).decode('utf-8')
        
        parts = binary_data.split("11111111", 1)
        if len(parts) != 2:
            logger.error("Invalid metadata separator.")
            return
        metadata_binary, data_binary = parts
        
        metadata_bytes = bytearray(int(metadata_binary[i:i+8], 2) for i in range(0, len(metadata_binary), 8))
        metadata_str = metadata_bytes.decode('utf-8')
        metadata = metadata_str.split("|")
        
        os.makedirs(output_folder, exist_ok=True)
        offset = 0
        for meta in metadata:
            if not meta:
                continue
            rel_path, size_str = meta.rsplit(":", 1)
            size = int(size_str)
            file_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file_binary = data_binary[offset:offset + size * 8]
            byte_data = bytearray(int(file_binary[i:i+8], 2) for i in range(0, len(file_binary), 8))
            with open(file_path, 'wb') as f:
                f.write(byte_data)
            offset += size * 8
            logger.info(f"Restored: {file_path}")
    except Exception as e:
        logger.error(f"Error restoring folder: {e}")

def write_to_storage_media(file_path, storage_path):
    """Write file to storage medium."""
    logger.info("Writing to storage medium...")
    try:
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        shutil.copy(file_path, storage_path)
        logger.info(f"Simulated writing to {storage_path}")
    except Exception as e:
        logger.error(f"Error writing to storage: {e}")

def derive_parameters_from_key(key):
    """Derive encryption parameters from key."""
    if not key:
        return 500, 700, 3
    hash_obj = hashlib.sha256(key.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    pitch1 = 300 + (int.from_bytes(hash_bytes[:4], 'big') % 500)
    pitch2 = 500 + (int.from_bytes(hash_bytes[4:8], 'big') % 500)
    num_tones = 2 + (int.from_bytes(hash_bytes[8:9], 'big') % 4)
    return pitch1, pitch2, num_tones

def validate_input_folder(folder_path):
    """Validate folder path."""
    if not os.path.isdir(folder_path):
        logger.error("Invalid folder path.")
        return False
    return True

def validate_noise_level(noise_level):
    """Validate noise level."""
    try:
        noise_level = float(noise_level)
        if not 0.0 <= noise_level <= 1.0:
            logger.error("Noise level must be between 0.0 and 1.0.")
            return None
        return noise_level
    except ValueError:
        logger.error("Noise level must be a number.")
        return None

def interactive_mode():
    """Run ToneCrypt in interactive mode."""
    print(f"{Fore.CYAN}Welcome to ToneCrypt v{VERSION}: Audio-based folder encryption tool{Style.RESET_ALL}")
    mode = input(f"{Fore.YELLOW}Choose mode (encrypt/decrypt): {Style.RESET_ALL}").lower()
    
    if mode == 'encrypt':
        folder_path = input(f"{Fore.YELLOW}Enter folder path to encrypt: {Style.RESET_ALL}")
        if not validate_input_folder(folder_path):
            return
        output_path = input(f"{Fore.YELLOW}Enter output file path (default: encrypted.wav): {Style.RESET_ALL}") or "encrypted.wav"
        storage_path = input(f"{Fore.YELLOW}Enter storage path (default: storage/{output_path}): {Style.RESET_ALL}") or f"storage/{output_path}"
        key = input(f"{Fore.YELLOW}Enter encryption key (or press Enter for defaults): {Style.RESET_ALL}")
        noise_level = input(f"{Fore.YELLOW}Enter noise level (0.0-1.0, default: 0.1): {Style.RESET_ALL}") or 0.1
        noise_level = validate_noise_level(noise_level)
        if noise_level is None:
            return
        
        pitch1, pitch2, num_tones = derive_parameters_from_key(key)
        temp_wav = "temp.wav"
        segment_length, tone_length = encrypt_folder(folder_path, temp_wav, storage_path, pitch1, pitch2, num_tones, noise_level)
        if segment_length and tone_length:
            os.rename(temp_wav, output_path)
            print(f"{Fore.GREEN}For decryption, use segment length: {segment_length}, tone length: {tone_length}, num tones: {num_tones}{Style.RESET_ALL}")
    
    elif mode == 'decrypt':
        source = input(f"{Fore.YELLOW}Choose decryption source (file/mic/cd): {Style.RESET_ALL}").lower()
        if source == 'file':
            input_path = input(f"{Fore.YELLOW}Enter input audio file path: {Style.RESET_ALL}")
            if not os.path.isfile(input_path):
                logger.error("Invalid audio file.")
                return
        elif source == 'mic':
            input_path = record_from_mic("recorded.wav")
            if not input_path:
                return
        elif source == 'cd':
            cd_drive = detect_cd_drive()
            if not cd_drive:
                return
            input_path = rip_from_cd("ripped.wav", cd_drive)
            if not input_path:
                return
        else:
            logger.error("Invalid source. Use 'file', 'mic', or 'cd'.")
            return
        
        input_wav = convert_to_wav(input_path)
        if not input_wav:
            return
        output_folder = input(f"{Fore.YELLOW}Enter output folder path (default: restored_folder): {Style.RESET_ALL}") or "restored_folder"
        try:
            segment_length = int(input(f"{Fore.YELLOW}Enter segment length (from encryption): {Style.RESET_ALL}"))
            tone_length = int(input(f"{Fore.YELLOW}Enter tone length (from encryption): {Style.RESET_ALL}"))
            num_tones = int(input(f"{Fore.YELLOW}Enter number of tones (default: 3): {Style.RESET_ALL}") or 3)
        except ValueError:
            logger.error("Segment length, tone length, and num tones must be integers.")
            return
        
        decrypt_folder(input_wav, output_folder, segment_length, tone_length, num_tones)
        if input_wav != input_path:
            os.remove(input_wav)

def encrypt_folder(folder_path, output_wav_path, storage_path, pitch_shift_hz1, pitch_shift_hz2, num_tones=3, noise_level=0.1):
    """Encrypt folder to WAV."""
    if not validate_input_folder(folder_path):
        return None, None
    
    binary_data, _ = folder_to_binary(folder_path)
    if not binary_data:
        return None, None
    
    wav_path, segment_length, tone_length = binary_to_wav(binary_data, output_wav_path, pitch_shift_hz1, pitch_shift_hz2, num_tones, noise_level=noise_level)
    if not wav_path:
        return None, None
    
    encrypted_path = encrypt_wav_with_pitch(wav_path, output_wav_path, pitch_shift_hz1, pitch_shift_hz2, segment_length, tone_length, num_tones)
    if not encrypted_path:
        return None, None
    
    write_to_storage_media(encrypted_path, storage_path)
    
    logger.info(f"Encryption complete! File saved as {output_wav_path}")
    return segment_length, tone_length

def decrypt_folder(input_wav_path, output_folder, segment_length, tone_length, num_tones=3):
    """Decrypt WAV to folder."""
    temp_wav = "decrypted_temp.wav"
    decrypted_wav = decrypt_wav_with_pitch(input_wav_path, temp_wav, segment_length, tone_length, num_tones)
    if not decrypted_wav:
        return
    
    logger.info("Converting WAV back to binary...")
    try:
        sample_rate, audio_data = wavfile.read(decrypted_wav)
        binary_data = ""
        duration_per_bit = 0.001
        samples_per_bit = int(sample_rate * duration_per_bit)
        
        threshold = np.mean(np.abs(audio_data)) * 0.6
        for i in range(0, len(audio_data), samples_per_bit):
            segment = audio_data[i:i+samples_per_bit]
            if len(segment) == 0:
                break
            avg_amplitude = np.mean(np.abs(segment))
            binary_data += '1' if avg_amplitude > threshold else '0'
        
        binary_to_folder(binary_data, output_folder)
        
        os.remove(temp_wav)
        logger.info(f"Decryption complete! Folder restored to {output_folder}")
    except Exception as e:
        logger.error(f"Error decrypting folder: {e}")

def main():
    """Main function for CLI and interactive mode."""
    parser = argparse.ArgumentParser(description=f"ToneCrypt v{VERSION}: Audio-based folder encryption/decryption")
    parser.add_argument('--version', action='version', version=f'ToneCrypt {VERSION}')
    subparsers = parser.add_subparsers(dest='command')

    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a folder to WAV')
    encrypt_parser.add_argument('--folder', help='Path to folder to encrypt')
    encrypt_parser.add_argument('--output', default='encrypted.wav', help='Output WAV file path')
    encrypt_parser.add_argument('--storage', help='Storage medium path (default: storage/<output>)')
    encrypt_parser.add_argument('--key', help='Encryption key')
    encrypt_parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level (0.0 to 1.0)')

    decrypt_parser = subparsers.add_parser('decrypt', help='Decrypt a WAV file to folder')
    decrypt_parser.add_argument('--input', help='Input WAV file path')
    decrypt_parser.add_argument('--output', default='restored_folder', help='Output folder path')
    decrypt_parser.add_argument('--segment-length', type=int, help='Length of first segment')
    decrypt_parser.add_argument('--tone-length', type=int, help='Length of all tones')
    decrypt_parser.add_argument('--num-tones', type=int, default=3, help='Number of tones')

    args = parser.parse_args()

    if not args.command:
        interactive_mode()
        return

    if args.command == 'encrypt':
        folder_path = args.folder
        output_wav = args.output
        storage_path = args.storage or f"storage/{output_wav}"
        noise_level = validate_noise_level(args.noise_level)
        if noise_level is None:
            return
        if not folder_path or not validate_input_folder(folder_path):
            return
        pitch1, pitch2, num_tones = derive_parameters_from_key(args.key)
        segment_length, tone_length = encrypt_folder(folder_path, output_wav, storage_path, pitch1, pitch2, num_tones, noise_level)
        if segment_length and tone_length:
            print(f"{Fore.GREEN}For decryption, use: --segment-length {segment_length} --tone-length {tone_length} --num-tones {num_tones}{Style.RESET_ALL}")
    elif args.command == 'decrypt':
        input_wav = convert_to_wav(args.input)
        if not input_wav:
            return
        if not args.segment_length or not args.tone_length:
            logger.error("Missing segment-length or tone-length.")
            return
        decrypt_folder(input_wav, args.output, args.segment_length, args.tone_length, args.num_tones)
        if input_wav != args.input:
            os.remove(input_wav)

if __name__ == "__main__":
    main()
