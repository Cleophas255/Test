import cv2
import numpy as np
import math
import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
import sys

TARGET_ADDRESS = "1crypto24HCr17BiMcKd5iVi5D4rsg1nK"
BITS = 256

# ---------------- Bitcoin helpers ----------------

def hash160(b):
    return hashlib.new('ripemd160', hashlib.sha256(b).digest()).digest()

def pubkey_to_address(pubkey_bytes):
    vh160 = b'\x00' + hash160(pubkey_bytes)
    checksum = hashlib.sha256(hashlib.sha256(vh160).digest()).digest()[:4]
    return base58.b58encode(vh160 + checksum).decode()

def check_key(hexkey):
    priv = bytes.fromhex(hexkey)
    sk = SigningKey.from_string(priv, curve=SECP256k1)
    vk = sk.verifying_key

    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()

    pub_u = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
    pub_c = (b'\x02' if y % 2 == 0 else b'\x03') + x.to_bytes(32, 'big')

    a1 = pubkey_to_address(pub_u)
    a2 = pubkey_to_address(pub_c)

    print("Uncompressed:", a1)
    print("Compressed  :", a2)

    if TARGET_ADDRESS in (a1, a2):
        print("\n🔥 SOLVED 🔥")

# ---------------- Core decoding ----------------

def extract_heights(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    h, w = bw.shape
    slice_w = w // BITS

    heights = []

    for i in range(BITS):
        x0 = i * slice_w
        x1 = min(w, (i + 1) * slice_w)

        slice_ = bw[:, x0:x1]
        ys = np.where(slice_ > 0)[0]

        heights.append(ys.max() - ys.min() if len(ys) > 0 else 0)

    return np.array(heights)

def estimate_decay(heights):
    xs = np.arange(len(heights))
    h = np.clip(heights, 1, None)

    logh = np.log(h)
    A = np.vstack([xs, np.ones(len(xs))]).T
    k, _ = np.linalg.lstsq(A, logh, rcond=None)[0]
    return k

def decode_bits(heights, k):
    xs = np.arange(len(heights))
    corrected = heights * np.exp(-k * xs)
    thresh = np.median(corrected)
    bits = ['1' if h > thresh else '0' for h in corrected]
    return bits

# ---------------- Main ----------------

def main(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Image load failed")

    heights = extract_heights(img)
    print("Samples:", len(heights))

    k = estimate_decay(heights)
    print("Decay k:", k)

    bits = decode_bits(heights, k)
    bitstring = ''.join(bits)

    print("First 64 bits:", bitstring[:64])

    hexkey = hex(int(bitstring, 2))[2:].zfill(64)
    print("\nPrivate key:", hexkey)

    check_key(hexkey)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halving_fixed_sampling.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])
