# ToneCrypt

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.3.1-blue.svg)

**ToneCrypt** is a Python-based tool that encrypts folders into WAV audio files using pitch shifts, noise, and reverb, with tones encoding decryption keys via a nonlinear equation `[cos(f1/200)*f2^3 + log(f3+1)*exp(f2/500) + sin(f1/f3)]`. It supports decryption from audio files, microphone recordings, or CDs, with Reed-Solomon error correction for robust folder restoration. This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## Features

- **Audio-Based Encryption**: Convert folders to WAV files with customizable pitch shifts and noise levels.
- **Flexible Decryption**: Decrypt from WAV files, microphone input, or CD audio.
- **Secure Key Derivation**: Uses SHA-256 to derive encryption parameters from a user-provided key.
- **Error Correction**: Implements Reed-Solomon encoding for reliable data recovery.
- **Interactive & CLI Modes**: Supports both interactive prompts and command-line arguments.

---

## Installation

### Option 1: Install via `.deb` Package 

1. **Download the Package**:
   Download `tonecrypt_1.3-1.deb` from the [Releases](https://github.com/yourusername/tonecrypt/releases) page.

2. **Install System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip libsndfile1 ffmpeg wodim cdparanoia alsa-utils
   ```

3. **Install the Package**:
   ```bash
   sudo apt install ./tonecrypt_1.3-1.deb
   ```

4. **Verify Installation**:
   ```bash
   tonecrypt --version
   ```
   Expected output: `ToneCrypt 1.3.1`

### Option 2: Install as a Python Package

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ewarggg776/tonecrypt.git
   cd tonecrypt
   ```

2. **Install System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip libsndfile1 ffmpeg wodim cdparanoia alsa-utils
   ```

3. **Install Python Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Run Directly**:
   ```bash
   python3 tonecrypt.py
   ```

---

## Usage

### Interactive Mode
Run `tonecrypt` without arguments to enter interactive mode:
```bash
tonecrypt
```
Follow the prompts to encrypt or decrypt a folder. Example:
```
Welcome to ToneCrypt v1.3.1: Audio-based folder encryption tool
Choose mode (encrypt/decrypt): encrypt
Enter folder path to encrypt: /home/user/test_folder
Enter output file path (default: encrypted.wav): encrypted.wav
Enter storage path (default: storage/encrypted.wav): /home/user/Music/tonecrypt
Enter encryption key (or press Enter for defaults): mysecret
Enter noise level (0.0-1.0, default: 0.1): 0.5
```

### Command-Line Mode
- **Encrypt a Folder**:
  ```bash
  tonecrypt encrypt --folder my_folder --key mysecret --output encrypted.wav --storage /path/to/storage --noise-level 0.5
  ```
  Note the `segment-length`, `tone-length`, and `num-tones` output for decryption.

- **Decrypt a WAV File**:
  ```bash
  tonecrypt decrypt --input encrypted.wav --output restored_folder --segment-length 123456 --tone-length 66150 --num-tones 3
  ```

### Examples
1. **Encrypt a Folder**:
   ```bash
   mkdir -p test_folder/subdir
   echo "Hello" > test_folder/test.txt
   echo "World" > test_folder/subdir/subtest.txt
   tonecrypt encrypt --folder test_folder --key mysecret
   ```

2. **Decrypt a WAV File**:
   ```bash
   tonecrypt decrypt --input encrypted.wav --segment-length 123456 --tone-length 66150
   ```

3. **Decrypt from Microphone**:
   ```bash
   tonecrypt
   ```
   Choose `decrypt`, select `mic`, and follow prompts.

---

## Building the `.deb` Package

To build the `.deb` package locally:
```bash
cd tonecrypt
./build-deb.sh
```
This creates `tonecrypt_1.3-1.deb` in the current directory. Install it as shown in the Installation section.

---

## Dependencies

| Type         | Dependencies                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **System**   | `python3 (>= 3.7)`, `python3-pip`, `libsndfile1`, `ffmpeg`, `wodim`, `cdparanoia`, `alsa-utils` |
| **Python**   | `numpy`, `scipy`, `soundfile`, `librosa`, `reedsolo`, `colorama`, `pydub` |

Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE). You are free to use, modify, and distribute this software, provided that any derivative works are also licensed under GPL-3.0.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please report issues or suggest improvements via the [Issues](https://github.com/ewarggg776/tonecrypt/issues) page.

---

## Troubleshooting

- **APT Lock Errors**:
  If you encounter `Could not get lock /var/lib/dpkg/lock-frontend`:
  ```bash
  sudo rm /var/lib/dpkg/lock-frontend
  sudo dpkg --configure -a
  sudo apt-get install -f
  ```

- **Command Not Found**:
  Ensure `/usr/local/bin` is in your PATH:
  ```bash
  echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
  source ~/.bashrc
  ```

- **Python Dependency Errors**:
  Reinstall dependencies:
  ```bash
  pip3 install --break-system-packages numpy scipy soundfile librosa reedsolo colorama pydub
  ```

For further assistance, open an issue on GitHub.
