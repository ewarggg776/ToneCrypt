#!/usr/bin/env python3

import os
import sys
import numpy as np
import scipy.signal
import librosa
import soundfile as sf
from reedsolo import RSCodec
from colorama import Fore, Style, init
import pydub
import argparse
import logging
import hashlib
import glob

# Initialize colorama for colored output
init()

# Custom pre_emphasis function to replace librosa.effects.pre_emphasis
def pre_emphasis(y, coef=0.97):
    return scipy.signal.lfilter([1, -coef], [1], y)

# Setup logging (default INFO, can be set to DEBUG with -V)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def folder_to_binary(folder_path):
    """Convert all files in a folder (recursively) to a bytes object with metadata."""
    logger.info("Converting folder to binary (with metadata)...")
    file_entries = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, folder_path)
            with open(fpath, 'rb') as f:
                file_bytes = f.read()
            # Store as: <len(path)>:<path><len(data)>:<data>
            entry = f"{len(relpath):08d}:{relpath}:{len(file_bytes):016d}:".encode('utf-8') + file_bytes
            file_entries.append(entry)
            logger.info(f"Added file: {relpath} ({len(file_bytes)} bytes)")
    all_bytes = b''.join(file_entries)
    return all_bytes

def binary_to_folder(binary_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if not isinstance(binary_data, (bytes, bytearray)):
        logger.error("Input to binary_to_folder is not valid bytes. Aborting file restoration.")
        return
    byte_data = binary_data
    idx = 0
    total = len(byte_data)
    restored = 0
    while idx < total:
        # Skip any non-digit bytes (e.g., newlines) before path length
        while idx < total and not (48 <= byte_data[idx] <= 57):  # ASCII '0'-'9'
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping non-digit byte at idx={idx}: {byte_data[idx:idx+1]!r}")
            idx += 1
        if idx + 8 > total:
            logger.error(f"Unexpected end of data while reading path length at index {idx}. Data may be truncated or corrupted.")
            break
        path_len_bytes = byte_data[idx:idx+8]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"idx={idx} path_len_bytes={path_len_bytes!r} (as str: {path_len_bytes.decode('utf-8', errors='replace')})")
        try:
            path_len = int(path_len_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to parse path length at index {idx}: {e}")
            break
        idx += 8
        # Skip the colon after path length
        if idx < total and byte_data[idx:idx+1] == b':':
            idx += 1
        else:
            logger.error(f"Expected colon after path length at index {idx}, found: {byte_data[idx:idx+1]!r}")
            break
        if idx + path_len > total:
            logger.error(f"Unexpected end of data while reading path at index {idx}. Data may be truncated or corrupted.")
            break
        relpath_bytes = byte_data[idx:idx+path_len]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"idx={idx} relpath_bytes={relpath_bytes!r} (as str: {relpath_bytes.decode('utf-8', errors='replace')})")
        relpath = relpath_bytes.decode('utf-8')
        idx += path_len
        # Skip the colon after relpath
        if idx < total and byte_data[idx:idx+1] == b':':
            idx += 1
        else:
            logger.error(f"Expected colon after relpath at index {idx}, found: {byte_data[idx:idx+1]!r}")
            break
        # Check for enough bytes for data length
        if idx + 16 > total:
            logger.error(f"Unexpected end of data while reading data length at index {idx}. Data may be truncated or corrupted.")
            break
        data_len_bytes = byte_data[idx:idx+16]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"idx={idx} data_len_bytes={data_len_bytes!r} (as str: {data_len_bytes.decode('utf-8', errors='replace')})")
        try:
            data_len = int(data_len_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to parse data length at index {idx}: {e}")
            break
        idx += 16
        if idx + data_len > total:
            logger.error(f"Unexpected end of data while reading file bytes at index {idx}. Data may be truncated or corrupted.")
            break
        file_bytes = byte_data[idx:idx+data_len]
        idx += data_len
        out_path = os.path.join(output_folder, relpath)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(file_bytes)
        logger.info(f"Restored file: {relpath} ({data_len} bytes)")
        restored += 1
    logger.info(f"Decryption complete. Restored {restored} files to {output_folder}")

def encode_binary_to_wav(binary_data, output_path, key, noise_level=0.1, num_tones=4, segment_length=1024, tone_length=256):
    """Encode binary data into a WAV file using multiple tones."""
    logger.info(f"Encoding binary to WAV with {num_tones} tones...")
    rsc = RSCodec(10)  # Reed-Solomon error correction
    # Encode and get bytes
    encoded_bytes = rsc.encode(binary_data)
    if logger.isEnabledFor(logging.DEBUG):
        with open('original_encoded_bytes.bin', 'wb') as f:
            f.write(encoded_bytes[:128])
        logger.debug(f"First 64 bytes of encoded_bytes: {list(encoded_bytes[:64])}")
    # Convert bytes to bit string
    bit_str = ''.join(f'{byte:08b}' for byte in encoded_bytes)

    # Group bits according to number of tones (e.g., 2 bits per tone for 4 tones)
    bits_per_tone = int(np.log2(num_tones))
    tone_indices = []
    for i in range(0, len(bit_str), bits_per_tone):
        group = bit_str[i:i+bits_per_tone]
        if len(group) < bits_per_tone:
            group = group.ljust(bits_per_tone, '0')  # pad last group
        tone_indices.append(int(group, 2))

    # Generate key-based seed for reproducibility
    key_hash = hashlib.sha256(key.encode()).digest()
    np.random.seed(int.from_bytes(key_hash[:4], 'big'))

    sample_rate = 44100
    duration = tone_length / sample_rate
    frequencies = np.linspace(1000, 5000, num_tones)
    audio_signal = np.zeros(len(tone_indices) * segment_length)

    for i, tone_idx in enumerate(tone_indices):
        segment = np.zeros(segment_length)
        for j in range(tone_length):
            t = j / sample_rate
            segment[j] += 0.5 * np.sin(2 * np.pi * frequencies[tone_idx] * t)
        audio_signal[i * segment_length:(i + 1) * segment_length] = segment
        if logger.isEnabledFor(logging.DEBUG) and (i % 10 == 0 or i == len(tone_indices) - 1):
            logger.debug(f"Encoded {i+1}/{len(tone_indices)} segments...")

    # Apply pre-emphasis filter
    audio_signal = pre_emphasis(audio_signal, coef=0.97)

    # Save clean signal for reference (not written to file)
    clean_signal = np.copy(audio_signal)

    # Add noise after main signal is generated
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(audio_signal))
        audio_signal += noise
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Noise added with std={noise_level}")

    # Normalize
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    # Save to WAV
    sf.write(output_path, audio_signal, sample_rate)
    logger.info(f"Encryption complete. WAV file saved at {output_path}")

def decode_wav_to_binary(wav_path, key, num_tones=4, segment_length=1024, tone_length=256):
    """Decode a WAV file back to binary data using multiple tones."""
    logger.info(f"Decoding WAV file {wav_path}...")
    # Generate key-based seed for reproducibility
    key_hash = hashlib.sha256(key.encode()).digest()
    np.random.seed(int.from_bytes(key_hash[:4], 'big'))
    sample_rate = 44100
    frequencies = np.linspace(1000, 5000, num_tones)
    audio_signal, _ = librosa.load(wav_path, sr=sample_rate)
    # Attempt to remove noise by smoothing (simple low-pass filter)
    if np.std(audio_signal) > 0.15:  # Heuristic: if noisy, smooth
        audio_signal = scipy.signal.medfilt(audio_signal, kernel_size=5)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Noise detected and reduced with median filter.")
    # Remove pre-emphasis
    def de_emphasis(y, coef=0.97):
        x = np.zeros_like(y)
        x[0] = y[0]
        for n in range(1, len(y)):
            x[n] = coef * x[n-1] + y[n]
        return x
    audio_signal = de_emphasis(audio_signal, coef=0.97)
    bits_per_tone = int(np.log2(num_tones))
    tone_indices = []
    debug_freqs = []
    for i in range(0, len(audio_signal), segment_length):
        segment = audio_signal[i:i+tone_length]
        if len(segment) < tone_length:
            break
        fft = np.fft.fft(segment)
        mag = np.abs(fft)
        freqs = np.fft.fftfreq(tone_length, 1/sample_rate)
        # Find all peaks in the magnitude spectrum
        peaks, _ = scipy.signal.find_peaks(mag, height=np.max(mag)*0.5)
        if len(peaks) == 0:
            idx = np.argmax(mag)
        else:
            idx = peaks[np.argmax(mag[peaks])]
        freq = abs(freqs[idx])
        debug_freqs.append(freq)
        # Average over a small window around the peak for robustness
        window = mag[max(0, idx-2):min(len(mag), idx+3)]
        avg_idx = np.argmax(window) + max(0, idx-2)
        freq = abs(freqs[avg_idx])
        tone_idx = int(np.argmin(np.abs(frequencies - freq)))
        tone_indices.append(tone_idx)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"First 20 detected frequencies: {debug_freqs[:20]}")
    # Convert tone indices to bit string
    bit_str = ''.join(f'{idx:0{bits_per_tone}b}' for idx in tone_indices)
    # Group bits into bytes
    byte_list = []
    for i in range(0, len(bit_str), 8):
        byte = bit_str[i:i+8]
        if len(byte) == 8:
            byte_list.append(int(byte, 2))
    byte_data = bytes(byte_list)
    # Diagnostics: print first bits/bytes
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"First 64 bits: {bit_str[:64]}")
        logger.debug(f"First 16 bytes: {list(byte_data[:16])}")
    # Try to decode with Reed-Solomon, and fallback to partial recovery if possible
    try:
        rsc = RSCodec(10)
        decoded = rsc.decode(byte_data)
        # reedsolo.decode may return (decoded_bytes, ecc) tuple
        if isinstance(decoded, tuple):
            decoded_bytes = decoded[0]
        else:
            decoded_bytes = decoded
        # Diagnostics: log and save the length and first 128 bytes
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Length of decoded_bytes after Reed-Solomon: {len(decoded_bytes)} bytes")
            logger.debug(f"First 64 bytes of decoded_bytes: {list(decoded_bytes[:64])}")
            with open('step2_decoded_bytes.bin', 'wb') as f:
                f.write(decoded_bytes[:128])
            logger.debug("Reed-Solomon decoding successful.")
        return decoded_bytes
    except Exception as e:
        logger.error(f"Error in Reed-Solomon decoding: {e}")
        # Save the raw bitstream for manual inspection
        debug_path = os.path.splitext(wav_path)[0] + "_recovered_bits.bin"
        with open(debug_path, "wb") as f:
            f.write(byte_data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.warning(f"Raw recovered bitstream saved to {debug_path}")
        # Try to recover as much as possible (best effort)
        try:
            decoded = byte_data.decode('ascii', errors='ignore')
            logger.warning("Partial recovery: output may be incomplete or corrupted.")
            return decoded
        except Exception as e2:
            logger.error(f"Partial recovery failed: {e2}")
    try:
        rsc = RSCodec(10)
        decoded = rsc.decode(byte_data)
        # reedsolo.decode may return (decoded_bytes, ecc) tuple
        if isinstance(decoded, tuple):
            decoded_bytes = decoded[0]
        else:
            decoded_bytes = decoded
        logger.info("Reed-Solomon decoding successful.")
        return decoded_bytes
    except Exception as e:
        logger.error(f"Error in Reed-Solomon decoding: {e}")
        # Save the raw bitstream for manual inspection
        debug_path = os.path.splitext(wav_path)[0] + "_recovered_bits.bin"
        with open(debug_path, "wb") as f:
            f.write(byte_data)
        logger.warning(f"Raw recovered bitstream saved to {debug_path}")
        # Try to recover as much as possible (best effort)
        try:
            # Only keep ASCII '0' and '1' to form a valid binary string
            if isinstance(byte_data, tuple):
                raw_bytes = byte_data[0]
            else:
                raw_bytes = byte_data
            decoded_str = raw_bytes.decode('ascii', errors='ignore')
            binary_str = ''.join(c for c in decoded_str if c in '01')
            if not binary_str or len(binary_str) < 8:
                logger.error("Partial recovery failed: not enough valid binary data.")
                return None
            logger.warning("Partial recovery: output may be incomplete or corrupted.")
            return binary_str
        except Exception as e2:
            logger.error(f"Partial recovery failed: {e2}")
            return None

def encrypt_folder(folder_path, output_path, storage_path, key, noise_level):
    """Encrypt a folder into a WAV file."""
    binary_data = folder_to_binary(folder_path)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"First 128 bytes of folder_to_binary: {binary_data[:128]}")
        logger.debug(f"Length of folder_to_binary output: {len(binary_data)} bytes")
        with open('step1_folder_to_binary.bin', 'wb') as f:
            f.write(binary_data[:1024])
    encode_binary_to_wav(binary_data, output_path, key, noise_level)
    os.makedirs(storage_path, exist_ok=True)
    logger.info(f"Storage saved at {storage_path}")
    logger.info(f"Decryption parameters: segment-length=1024, tone-length=256, num-tones=4")

def main():
    parser = argparse.ArgumentParser(description="ToneCrypt: Encrypt/decrypt files into audio tones")
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], help="Operation mode")
    parser.add_argument('--input', help="Input folder (encrypt) or WAV file (decrypt)")
    parser.add_argument('--output', default='encrypted.wav', help="Output WAV file (encrypt) or folder (decrypt)")
    parser.add_argument('--storage', default='storage', help="Storage path for encrypted data")
    parser.add_argument('--key', default='default_key', help="Encryption/decryption key")
    parser.add_argument('--noise', type=float, default=0.1, help="Noise level (0.0-1.0)")
    parser.add_argument('-V', '--verbose', action='store_true', help="Enable verbose (debug) output")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)


    # Interactive mode if --mode is not provided
    if not args.mode:
        from colorama import Fore, Style
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== ToneCrypt Interactive Mode ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}1){Style.RESET_ALL} {Fore.GREEN}Encrypt a folder to WAV (default){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}2){Style.RESET_ALL} {Fore.MAGENTA}Decrypt a WAV file to folder{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}V){Style.RESET_ALL} {Fore.BLUE}Toggle Verbose Mode{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Q){Style.RESET_ALL} Quit{Style.RESET_ALL}")
        verbose_mode = False
        while True:
            choice = input(f"{Fore.CYAN}Select an option [{Fore.YELLOW}1{Fore.CYAN}]: {Style.RESET_ALL}").strip().lower()
            if choice in ('', '1', 'encrypt'):
                args.mode = 'encrypt'
                break
            elif choice in ('2', 'decrypt'):
                args.mode = 'decrypt'
                break
            elif choice == 'v':
                verbose_mode = not verbose_mode
                if verbose_mode:
                    logger.setLevel(logging.DEBUG)
                    print(f"{Fore.BLUE}Verbose mode enabled.{Style.RESET_ALL}")
                else:
                    logger.setLevel(logging.INFO)
                    print(f"{Fore.BLUE}Verbose mode disabled.{Style.RESET_ALL}")
            elif choice == 'q':
                print(f"{Fore.RED}Exiting.{Style.RESET_ALL}")
                sys.exit(0)
            else:
                print(f"{Fore.RED}Invalid option. Please choose 1, 2, V, or Q.{Style.RESET_ALL}")

        def prompt(msg, default=None, required=False, color=Fore.WHITE):
            while True:
                val = input(f"{color}{msg}{Style.RESET_ALL}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
                if val:
                    return val
                elif default is not None:
                    return default
                elif required:
                    print(f"{Fore.RED}This field is required.{Style.RESET_ALL}")
                else:
                    return ''

        if args.mode == 'encrypt':
            args.input = args.input or prompt("Enter input folder to encrypt", required=True, color=Fore.YELLOW)
            args.output = prompt("Enter output WAV file", args.output, color=Fore.YELLOW)
            args.storage = prompt("Enter storage path", args.storage, color=Fore.YELLOW)
            args.key = prompt("Enter encryption key", args.key, color=Fore.YELLOW)
            while True:
                noise = prompt("Enter noise level (0.0-1.0)", str(args.noise), color=Fore.YELLOW)
                try:
                    args.noise = float(noise)
                    if 0.0 <= args.noise <= 1.0:
                        break
                    else:
                        print(f"{Fore.RED}Noise level must be between 0.0 and 1.0.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid noise level, please enter a number between 0.0 and 1.0.{Style.RESET_ALL}")
        elif args.mode == 'decrypt':
            args.input = args.input or prompt("Enter input WAV file to decrypt", required=True, color=Fore.MAGENTA)
            args.output = prompt("Enter output folder for restored files", args.output, color=Fore.MAGENTA)
            args.key = prompt("Enter decryption key", args.key, color=Fore.MAGENTA)

    if args.mode == 'encrypt':
        if not args.input or not os.path.isdir(args.input):
            logger.error(f"Input folder {args.input} does not exist")
            sys.exit(1)
        encrypt_folder(args.input, args.output, args.storage, args.key, args.noise)
    elif args.mode == 'decrypt':
        if not args.input or not os.path.isfile(args.input):
            logger.error(f"Input WAV file {args.input} does not exist")
            sys.exit(1)
        binary_data = decode_wav_to_binary(args.input, args.key)
        if not binary_data:
            logger.error("Decryption failed: could not recover valid data from audio. See _recovered_bits.bin for raw output.")
            sys.exit(1)
        # If the data is bytes, skip binary string check and pass directly
        if isinstance(binary_data, (bytes, bytearray)):
            binary_to_folder(binary_data, args.output)
        else:
            # If the data is a string, check if it's a valid binary string
            if not all(c in '01' for c in binary_data[:64]):
                logger.warning("Recovered data does not appear to be a valid binary string. Manual inspection may be required.")
            binary_to_folder(binary_data, args.output)
    else:
        logger.error("Unknown mode")
        sys.exit(1)

if __name__ == "__main__":
    main()