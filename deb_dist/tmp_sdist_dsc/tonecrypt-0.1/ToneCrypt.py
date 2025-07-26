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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def folder_to_binary(folder_path):
    """Convert all files in a folder to a binary string."""
    logger.info("Converting folder to binary...")
    binary_data = ""
    file_count = len(glob.glob(os.path.join(folder_path, '*')))
    processed = 0

    for filename in glob.glob(os.path.join(folder_path, '*')):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                file_data = f.read()
                binary_data += ''.join(format(byte, '08b') for byte in file_data)
            processed += 1
            logger.info(f"Processed file {processed}/{file_count}...")
    return binary_data

def encode_binary_to_wav(binary_data, output_path, key, noise_level=0.1, num_tones=4, segment_length=1024, tone_length=256):
    """Encode binary data into a WAV file using multiple tones."""
    logger.info(f"Encoding binary to WAV with {num_tones} tones...")
    rsc = RSCodec(10)  # Reed-Solomon error correction
    binary_data = rsc.encode(binary_data.encode('ascii')).decode('latin1')

    # Generate key-based seed for reproducibility
    key_hash = hashlib.sha256(key.encode()).digest()
    np.random.seed(int.from_bytes(key_hash[:4], 'big'))

    sample_rate = 44100
    duration = tone_length / sample_rate
    frequencies = np.linspace(1000, 5000, num_tones)
    audio_signal = np.zeros(len(binary_data) * segment_length)

    for i, bit_group in enumerate(binary_data):
        segment = np.zeros(segment_length)
        for j in range(tone_length):
            t = j / sample_rate
            tone_idx = int(bit_group, 2) % num_tones
            segment[j] += 0.5 * np.sin(2 * np.pi * frequencies[tone_idx] * t)
        audio_signal[i * segment_length:(i + 1) * segment_length] = segment
        logger.info(f"Encoded {i}/{len(binary_data)} bits (segment {i + 1})...")

    # Apply pre-emphasis filter
    audio_signal = pre_emphasis(audio_signal, coef=0.97)

    # Add noise
    noise = np.random.normal(0, noise_level, len(audio_signal))
    audio_signal += noise

    # Normalize
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    # Save to WAV
    sf.write(output_path, audio_signal, sample_rate)
    logger.info(f"Encryption complete. WAV file saved at {output_path}")

def encrypt_folder(folder_path, output_path, storage_path, key, noise_level):
    """Encrypt a folder into a WAV file."""
    binary_data = folder_to_binary(folder_path)
    encode_binary_to_wav(binary_data, output_path, key, noise_level)
    os.makedirs(storage_path, exist_ok=True)
    logger.info(f"Storage saved at {storage_path}")
    logger.info(f"Decryption parameters: segment-length={1024}, tone-length={256}, num-tones=4")

def main():

    parser = argparse.ArgumentParser(description="ToneCrypt: Encrypt/decrypt files into audio tones")
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], help="Operation mode")
    parser.add_argument('--input', help="Input folder (encrypt) or WAV file (decrypt)")
    parser.add_argument('--output', default='encrypted.wav', help="Output WAV file (encrypt) or folder (decrypt)")
    parser.add_argument('--storage', default='storage', help="Storage path for encrypted data")
    parser.add_argument('--key', default='default_key', help="Encryption/decryption key")
    parser.add_argument('--noise', type=float, default=0.1, help="Noise level (0.0-1.0)")

    args = parser.parse_args()

    # Improved Interactive mode if --mode is not provided
    if not args.mode:
        print("\n=== ToneCrypt Interactive Mode ===")
        print("1) Encrypt a folder to WAV (default)")
        print("2) Decrypt a WAV file to folder (not implemented)")
        print("q) Quit")
        while True:
            choice = input("Select an option [1]: ").strip().lower()
            if choice in ('', '1', 'encrypt'):
                args.mode = 'encrypt'
                break
            elif choice in ('2', 'decrypt'):
                print("Decryption mode not implemented in this version.")
                sys.exit(1)
            elif choice == 'q':
                print("Exiting.")
                sys.exit(0)
            else:
                print("Invalid option. Please choose 1, 2, or q.")

        # Prompt for encrypt options with defaults
        def prompt(msg, default=None, required=False):
            while True:
                val = input(f"{msg}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
                if val:
                    return val
                elif default is not None:
                    return default
                elif required:
                    print("This field is required.")
                else:
                    return ''

        args.input = args.input or prompt("Enter input folder to encrypt", required=True)
        args.output = prompt("Enter output WAV file", args.output)
        args.storage = prompt("Enter storage path", args.storage)
        args.key = prompt("Enter encryption key", args.key)
        while True:
            noise = prompt("Enter noise level (0.0-1.0)", str(args.noise))
            try:
                args.noise = float(noise)
                if 0.0 <= args.noise <= 1.0:
                    break
                else:
                    print("Noise level must be between 0.0 and 1.0.")
            except ValueError:
                print("Invalid noise level, please enter a number between 0.0 and 1.0.")

    if args.mode == 'encrypt':
        if not args.input or not os.path.isdir(args.input):
            logger.error(f"Input folder {args.input} does not exist")
            sys.exit(1)
        encrypt_folder(args.input, args.output, args.storage, args.key, args.noise)
    else:
        logger.error("Decryption mode not implemented in this version")
        sys.exit(1)

if __name__ == "__main__":
    main()