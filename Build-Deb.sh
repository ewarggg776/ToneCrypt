#!/bin/bash
set -e

# Get the directory containing this script (where tonecrypt.py is assumed to be)
BUILD_DIR=$(pwd)

# Create package structure
mkdir -p tonecrypt_1.3-1/DEBIAN
mkdir -p tonecrypt_1.3-1/usr/local/bin

# Copy tonecrypt.py to package
if [ ! -f "$BUILD_DIR/tonecrypt.py" ]; then
    echo "Error: tonecrypt.py not found in $BUILD_DIR"
    exit 1
fi
cp "$BUILD_DIR/tonecrypt.py" tonecrypt_1.3-1/usr/local/bin/tonecrypt.py
chmod 644 tonecrypt_1.3-1/usr/local/bin/tonecrypt.py

# Create DEBIAN/control
cat > tonecrypt_1.3-1/DEBIAN/control << 'EOF'
Package: tonecrypt
Version: 1.3-1
Section: utils
Priority: optional
Architecture: all
Depends: python3 (>= 3.7), python3-pip, libsndfile1, ffmpeg, wodim, cdparanoia, alsa-utils
Maintainer: Ewarggg776
Description: ToneCrypt: Audio-based folder encryption tool
 Encrypts folders into WAV files using pitch shifts, noise, and reverb, with tones encoding decryption keys via a nonlinear equation [cos(f1/200)*f2^3 + log(f3+1)*exp(f2/500) + sin(f1/f3)].
 Supports decryption from files, microphone, or CD, with error correction and folder restoration.
EOF

# Create DEBIAN/postinst
cat > tonecrypt_1.3-1/DEBIAN/postinst << 'EOF'
#!/bin/sh
set -e
pip3 install --break-system-packages numpy scipy soundfile librosa reedsolo colorama pydub
echo "Python dependencies installed successfully."
exit 0
EOF
chmod +x tonecrypt_1.3-1/DEBIAN/postinst

# Create wrapper script pointing to /usr/local/bin/tonecrypt.py
cat > tonecrypt_1.3-1/usr/local/bin/tonecrypt << 'EOF'
#!/bin/sh
python3 /usr/local/bin/tonecrypt.py "$@"
EOF
chmod +x tonecrypt_1.3-1/usr/local/bin/tonecrypt

# Build the .deb package
dpkg-deb --build tonecrypt_1.3-1
echo "Created tonecrypt_1.3-1.deb in $BUILD_DIR"
