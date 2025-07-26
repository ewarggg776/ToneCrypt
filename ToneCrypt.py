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
    # Specify which tone indices are reserved for noise (e.g., last one by default)
    noise_tones = [num_tones - 1]  # You can customize this list
    data_tones = [i for i in range(num_tones) if i not in noise_tones]
    rsc = RSCodec(32)  # Increased Reed-Solomon error correction
    encoded_bytes = rsc.encode(binary_data)
    if logger.isEnabledFor(logging.DEBUG):
        with open('original_encoded_bytes.bin', 'wb') as f:
            f.write(encoded_bytes[:128])
        logger.debug(f"First 64 bytes of encoded_bytes: {list(encoded_bytes[:64])}")
    # Convert bytes to bit string
    bits_per_tone = int(np.log2(len(data_tones)))
    bit_str = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    # Map bits to data_tones only
    tone_indices = []
    for i in range(0, len(bit_str), bits_per_tone):
        group = bit_str[i:i+bits_per_tone]
        if len(group) < bits_per_tone:
            group = group.ljust(bits_per_tone, '0')
        idx = int(group, 2)
        if idx >= len(data_tones):
            idx = 0
        tone_indices.append(data_tones[idx])
    # Header: first segment encodes noise_tones as a binary mask
    header = np.zeros(segment_length)
    sample_rate = 44100
    frequencies = np.linspace(1000, 5000, num_tones)
    t_vec = np.arange(tone_length) / sample_rate
    # Vectorized header generation
    for t in noise_tones:
        header[:tone_length] += 0.5 * np.sin(2 * np.pi * frequencies[t] * t_vec)
    # Main encoding: vectorized segment generation
    n_segments = len(tone_indices)
    audio_signal = np.zeros((n_segments + 1) * segment_length)
    audio_signal[:segment_length] = header
    # Precompute all tone segments
    tone_segments = 0.5 * np.sin(2 * np.pi * frequencies[:, None] * t_vec)
    for i, tone_idx in enumerate(tone_indices):
        segment = np.zeros(segment_length)
        segment[:tone_length] = tone_segments[tone_idx]
        audio_signal[(i+1) * segment_length:(i+2) * segment_length] = segment
        if logger.isEnabledFor(logging.DEBUG) and (i % 100 == 0 or i == n_segments - 1):
            logger.debug(f"Encoded {i+1}/{n_segments} segments...")
    audio_signal = pre_emphasis(audio_signal, coef=0.97)
    if noise_level > 0:
        audio_signal += np.random.normal(0, noise_level, len(audio_signal))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Noise added with std={noise_level}")
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    sf.write(output_path, audio_signal, sample_rate)
    logger.info(f"Encryption complete. WAV file saved at {output_path}")
    # If in verbose/debug mode, show the waveform graph
    if logger.isEnabledFor(logging.DEBUG):
        # ASCII waveform for first second (44100 samples), with / and \ connecting dots
        try:
            samples = audio_signal[:min(44100, len(audio_signal))]
            width = 80
            height = 16
            min_val, max_val = np.min(samples), np.max(samples)
            scale = (max_val - min_val) if (max_val - min_val) != 0 else 1
            points = []
            for x in range(width):
                idx = int(x * len(samples) / width)
                val = samples[idx]
                y = int((val - min_val) / scale * (height - 1))
                points.append(y)
            # Draw lines between points for a connected waveform
            rows = [[' ' for _ in range(width)] for _ in range(height)]
            for x in range(width):
                y = points[x]
                if x == 0:
                    rows[height - 1 - y][x] = '*'
                else:
                    prev_y = points[x-1]
                    if y == prev_y:
                        rows[height - 1 - y][x] = '-'
                    else:
                        # Draw vertical or diagonal connection
                        step = 1 if y > prev_y else -1
                        for interp_y in range(prev_y, y + step, step):
                            if interp_y == y:
                                # End point
                                if y > prev_y:
                                    rows[height - 1 - interp_y][x] = '\\'
                                else:
                                    rows[height - 1 - interp_y][x] = '/'
                            elif interp_y == prev_y:
                                # Start point
                                continue
                            else:
                                rows[height - 1 - interp_y][x] = '|'
            logger.info('ASCII waveform (first second):')
            for row in rows:
                logger.info(''.join(row))
            logger.info('ASCII waveform (first second):')
            for row in rows:
                logger.info(row)
        except Exception as e:
            logger.warning(f"Could not display ASCII waveform: {e}")

def decode_wav_to_binary(wav_path, key, num_tones=4, segment_length=1024, tone_length=256):
    """Decode a WAV file back to binary data using multiple tones."""
    logger.info(f"Decoding WAV file {wav_path}...")
    # Generate key-based seed for reproducibility
    key_hash = hashlib.sha256(key.encode()).digest()
    np.random.seed(int.from_bytes(key_hash[:4], 'big'))
    # Use 8 tones and increased RS redundancy by default
    num_tones = 8
    rsc = RSCodec(32)
    sample_rate = 44100
    frequencies = np.linspace(1000, 5000, num_tones)
    audio_signal, _ = librosa.load(wav_path, sr=sample_rate)
    # Read header to determine noise_tones
    header = audio_signal[:segment_length]
    detected_noise_tones = []
    for t in range(num_tones):
        # FFT energy in the header for each tone
        segment = header[:tone_length]
        fft = np.fft.fft(segment)
        mag = np.abs(fft)
        freqs = np.fft.fftfreq(tone_length, 1/sample_rate)
        tone_freq = frequencies[t]
        idx = np.argmin(np.abs(freqs - tone_freq))
        if mag[idx] > 5:  # Threshold for presence
            detected_noise_tones.append(t)
    data_tones = [i for i in range(num_tones) if i not in detected_noise_tones]
    bits_per_tone = int(np.log2(len(data_tones)))
    # Remove header from audio_signal
    audio_signal = audio_signal[segment_length:]
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
    tone_indices = []
    debug_freqs = []
    for i in range(0, len(audio_signal), segment_length):
        segment = audio_signal[i:i+tone_length]
        if len(segment) < tone_length:
            break
        fft = np.fft.fft(segment)
        mag = np.abs(fft)
        freqs = np.fft.fftfreq(tone_length, 1/sample_rate)
        peaks, _ = scipy.signal.find_peaks(mag, height=np.max(mag)*0.5)
        if len(peaks) == 0:
            idx = np.argmax(mag)
        else:
            idx = peaks[np.argmax(mag[peaks])]
        freq = abs(freqs[idx])
        debug_freqs.append(freq)
        window = mag[max(0, idx-2):min(len(mag), idx+3)]
        avg_idx = np.argmax(window) + max(0, idx-2)
        freq = abs(freqs[avg_idx])
        tone_idx = int(np.argmin(np.abs(frequencies - freq)))
        if tone_idx in data_tones:
            tone_indices.append(data_tones.index(tone_idx))
        # else: skip, as it's a noise tone
    # Convert tone indices to bit string using only data_tones
    bit_str = ''.join(f'{idx:0{bits_per_tone}b}' for idx in tone_indices)
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
    # If input is a .zip file, read as bytes; else, treat as folder
    if folder_path.lower().endswith('.zip') and os.path.isfile(folder_path):
        logger.info(f"Encrypting .zip file: {folder_path}")
        with open(folder_path, 'rb') as f:
            binary_data = f.read()
    else:
        binary_data = folder_to_binary(folder_path)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"First 128 bytes of input: {binary_data[:128]}")
        logger.debug(f"Length of input: {len(binary_data)} bytes")
        with open('step1_input_bytes.bin', 'wb') as f:
            f.write(binary_data[:1024])
    encode_binary_to_wav(binary_data, output_path, key, noise_level, num_tones=8)
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
        import curses
        from curses import wrapper
        menu_items = [
            ("Encrypt a folder to WAV (default)", "encrypt"),
            ("Decrypt a WAV file to folder", "decrypt"),
            ("Toggle Verbose Mode", "verbose"),
            ("Quit", "quit")
        ]
        verbose_mode = False
        def curses_menu(stdscr):
            import time
            curses.curs_set(0)  # Hide terminal cursor, use our own
            curses.start_color()
            has_256 = False
            try:
                if curses.COLORS >= 16:
                    has_256 = True
            except Exception:
                pass
            GREY = 8 if has_256 else curses.COLOR_WHITE
            curses.init_pair(10, curses.COLOR_BLACK, GREY)
            stdscr.bkgd(' ', curses.color_pair(10))
            curses.init_pair(1, curses.COLOR_WHITE, GREY)
            curses.init_pair(2, curses.COLOR_GREEN, GREY)
            curses.init_pair(3, curses.COLOR_RED, GREY)
            curses.init_pair(4, curses.COLOR_BLUE, GREY)
            curses.init_pair(5, curses.COLOR_RED, GREY)
            curses.init_pair(6, curses.COLOR_YELLOW, GREY)
            curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # For input field
            curses.init_pair(8, curses.COLOR_CYAN, GREY)  # For help text
            current_row = 0
            nonlocal verbose_mode
            blink = True
            last_blink = time.time()
            blink_interval = 0.5
            help_lines = [
                "Use UP/DOWN arrows to move, ENTER to select, ESC to quit.",
                "Home/End: jump to first/last. Toggle Verbose for debug info."
            ]
            while True:
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                min_h = 14
                min_w = 60
                if h < min_h or w < min_w:
                    stdscr.addstr(0, 0, "Terminal too small! Resize, then press any key.", curses.color_pair(5) | curses.A_BOLD)
                    stdscr.refresh()
                    stdscr.getch()
                    continue
                title = "=== ToneCrypt Interactive Mode ==="
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(1, w//2 - len(title)//2, title)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
                menu_x = w//2 - 25
                menu_y = h//2 - len(menu_items)
                for idx, (label, _) in enumerate(menu_items):
                    x = menu_x
                    y = menu_y + idx * 2
                    if idx == 0:
                        color = curses.color_pair(2)
                    elif idx == 1:
                        color = curses.color_pair(3)
                    elif idx == 2:
                        color = curses.color_pair(4)
                    elif idx == 3:
                        color = curses.color_pair(5)
                    if idx == current_row:
                        # Animated blinking arrow
                        if blink:
                            stdscr.attron(curses.color_pair(1) | curses.A_REVERSE | curses.A_BOLD)
                            stdscr.addstr(y, x, f" 7 {label} 1 ")
                            stdscr.attroff(curses.color_pair(1) | curses.A_REVERSE | curses.A_BOLD)
                        else:
                            stdscr.attron(curses.color_pair(1) | curses.A_REVERSE)
                            stdscr.addstr(y, x, f"   {label}   ")
                            stdscr.attroff(curses.color_pair(1) | curses.A_REVERSE)
                    else:
                        stdscr.attron(color)
                        stdscr.addstr(y, x, f"   {label}   ")
                        stdscr.attroff(color)
                # Show verbose mode status
                vstat = "ON" if verbose_mode else "OFF"
                stdscr.addstr(h-5, 2, f"Verbose Mode: {vstat}", curses.color_pair(4) if verbose_mode else curses.color_pair(6))
                # Help panel
                for i, line in enumerate(help_lines):
                    stdscr.addstr(h-3+i, 2, line, curses.color_pair(8) | curses.A_DIM)
                stdscr.refresh()
                # Blinking effect
                now = time.time()
                if now - last_blink > blink_interval:
                    blink = not blink
                    last_blink = now
                # Non-blocking input with timeout for animation
                stdscr.timeout(100)
                key = stdscr.getch()
                if key == -1:
                    continue
                if key in (curses.KEY_UP, ord('k')):
                    current_row = (current_row - 1) % len(menu_items)
                elif key in (curses.KEY_DOWN, ord('j')):
                    current_row = (current_row + 1) % len(menu_items)
                elif key in (curses.KEY_HOME, 262):
                    current_row = 0
                elif key in (curses.KEY_END, 360):
                    current_row = len(menu_items) - 1
                elif key in (27,):  # ESC
                    stdscr.clear()
                    stdscr.addstr(h//2, w//2 - 5, "Exiting...", curses.color_pair(5))
                    stdscr.refresh()
                    curses.napms(800)
                    sys.exit(0)
                elif key in (curses.KEY_ENTER, 10, 13):
                    label, action = menu_items[current_row]
                    if action == "encrypt":
                        return "encrypt"
                    elif action == "decrypt":
                        return "decrypt"
                    elif action == "verbose":
                        verbose_mode = not verbose_mode
                        if verbose_mode:
                            logger.setLevel(logging.DEBUG)
                        else:
                            logger.setLevel(logging.INFO)
                    elif action == "quit":
                        stdscr.clear()
                        stdscr.addstr(h//2, w//2 - 5, "Exiting...", curses.color_pair(5))
                        stdscr.refresh()
                        curses.napms(800)
                        sys.exit(0)
        # Prompt for input in curses
        def curses_prompt(stdscr, msg, default=None, required=False, color_pair=2):
            h, w = stdscr.getmaxyx()
            inp = ''
            error = ''
            cursor = 0
            while True:
                stdscr.clear()
                stdscr.attron(curses.color_pair(color_pair) | curses.A_BOLD)
                stdscr.addstr(h//2, w//2 - len(msg)//2, msg)
                stdscr.attroff(curses.color_pair(color_pair) | curses.A_BOLD)
                if default is not None:
                    stdscr.addstr(h//2+1, w//2 - 10, f"[Default: {default}]", curses.color_pair(8))
                # Input field with highlight
                field_x = w//2 - 20
                stdscr.attron(curses.color_pair(7))
                stdscr.addstr(h//2+2, field_x, ' ' * 40)
                stdscr.attroff(curses.color_pair(7))
                stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
                stdscr.addstr(h//2+2, field_x, inp + ' ' * (40 - len(inp)))
                stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)
                # Show blinking cursor
                stdscr.move(h//2+2, field_x + cursor)
                stdscr.refresh()
                # Error message
                if error:
                    stdscr.addstr(h//2+4, w//2 - len(error)//2, error, curses.color_pair(6) | curses.A_BOLD)
                key = stdscr.getch()
                if key in (curses.KEY_ENTER, 10, 13):
                    val = inp.strip()
                    if val:
                        return val
                    elif default is not None:
                        return default
                    elif required:
                        error = "This field is required!"
                        continue
                    else:
                        return ''
                elif key in (27,):  # ESC
                    return ''
                elif key in (curses.KEY_LEFT, 260):
                    cursor = max(0, cursor - 1)
                elif key in (curses.KEY_RIGHT, 261):
                    cursor = min(len(inp), cursor + 1)
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if cursor > 0:
                        inp = inp[:cursor-1] + inp[cursor:]
                        cursor -= 1
                elif key == curses.KEY_DC:
                    if cursor < len(inp):
                        inp = inp[:cursor] + inp[cursor+1:]
                elif 32 <= key <= 126:
                    if len(inp) < 40:
                        inp = inp[:cursor] + chr(key) + inp[cursor:]
                        cursor += 1
                else:
                    pass
        # Run menu
        mode = wrapper(curses_menu)
        # Prompt for parameters
        def run_curses_prompts(mode):
            return wrapper(lambda stdscr: _curses_prompts(stdscr, mode))
        def _curses_prompts(stdscr, mode):
            if mode == 'encrypt':
                args_input = curses_prompt(stdscr, "Enter input folder to encrypt", required=True, color_pair=3)
                args_output = curses_prompt(stdscr, "Enter output WAV file", "encrypted.wav", color_pair=3)
                args_storage = curses_prompt(stdscr, "Enter storage path", "storage", color_pair=3)
                args_key = curses_prompt(stdscr, "Enter encryption key", "default_key", color_pair=3)
                while True:
                    noise = curses_prompt(stdscr, "Enter noise level (0.0-1.0)", str(args.noise), color_pair=3)
                    try:
                        args_noise = float(noise)
                        if 0.0 <= args_noise <= 1.0:
                            break
                        else:
                            stdscr.addstr(2, 2, "Noise level must be between 0.0 and 1.0!", curses.color_pair(6))
                            stdscr.refresh()
                            curses.napms(1000)
                    except ValueError:
                        stdscr.addstr(2, 2, "Invalid noise level!", curses.color_pair(6))
                        stdscr.refresh()
                        curses.napms(1000)
                return {'input': args_input, 'output': args_output, 'storage': args_storage, 'key': args_key, 'noise': args_noise, 'mode': 'encrypt'}
            elif mode == 'decrypt':
                args_input = curses_prompt(stdscr, "Enter input WAV file to decrypt", required=True, color_pair=4)
                args_output = curses_prompt(stdscr, "Enter output folder for restored files", "decrypted", color_pair=4)
                args_key = curses_prompt(stdscr, "Enter decryption key", "default_key", color_pair=4)
                return {'input': args_input, 'output': args_output, 'key': args_key, 'mode': 'decrypt'}
        params = run_curses_prompts(mode)
        args.input = params.get('input', args.input)
        args.output = params.get('output', args.output)
        args.storage = params.get('storage', args.storage)
        args.key = params.get('key', args.key)
        args.noise = params.get('noise', args.noise)
        args.mode = params['mode']

    if args.mode == 'encrypt':
        # Accept .zip file or folder as input
        if not args.input or (not os.path.isdir(args.input) and not (args.input.lower().endswith('.zip') and os.path.isfile(args.input))):
            logger.error(f"Input folder or .zip file {args.input} does not exist")
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
        # If output ends with .zip, write bytes directly as a zip file
        if args.output.lower().endswith('.zip'):
            if not isinstance(binary_data, (bytes, bytearray)):
                logger.error("Decoded data is not bytes; cannot write .zip file.")
                sys.exit(1)
            with open(args.output, 'wb') as f:
                f.write(binary_data)
            logger.info(f"Decrypted .zip file written to {args.output}")
        else:
            # If the data is bytes, treat as folder archive
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