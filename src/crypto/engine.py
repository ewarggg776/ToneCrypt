import os
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np

class CryptoEngine:
    def __init__(self, password):
        self.password = password.encode()
        self.compressor = zstd.ZstdCompressor(level=15)
        self.decompressor = zstd.ZstdDecompressor()

    def _derive_key(self, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_data(self, data):
        """Compresses and then encrypts data using AES-256-GCM with a dynamic salt."""
        # 1. Generate a random 16-byte salt for this specific vault
        salt = os.urandom(16)
        key = self._derive_key(salt)
        
        # 2. Compress payload
        compressed = self.compressor.compress(data)
        
        # 3. Encrypt with AES-GCM
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        encrypted = aesgcm.encrypt(nonce, compressed, None)
        
        # 4. Return SALT + NONCE + CIPHERTEXT (Total header overhead: 28 bytes)
        return salt + nonce + encrypted

    def decrypt_data(self, data_blob):
        """Extracts salt, derives key, and decrypts/decompresses data."""
        if len(data_blob) < 28:
            raise ValueError("Data blob too small to contain salt and nonce.")
            
        # 1. Extract headers
        salt = data_blob[:16]
        nonce = data_blob[16:28]
        ciphertext = data_blob[28:]
        
        # 2. Derive key from the embedded salt
        key = self._derive_key(salt)
        
        # 3. Decrypt
        aesgcm = AESGCM(key)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        
        # 4. Decompress
        return self.decompressor.decompress(decrypted)

    def logistic_scramble(self, data_bytes, seed):
        """A custom nonlinear scrambling layer (optional obfuscation)."""
        data_np = np.frombuffer(data_bytes, dtype=np.uint8)
        r = 3.99 
        x = seed 
        
        scramble_seq = []
        for _ in range(len(data_np)):
            x = r * x * (1 - x)
            byte_val = int((x * 1000000) % 256)
            scramble_seq.append(byte_val)
        
        scramble_np = np.array(scramble_seq, dtype=np.uint8)
        return (data_np ^ scramble_np).tobytes()
