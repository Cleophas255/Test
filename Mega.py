import cv2
import numpy as np
import math
import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
import sys

TARGET_ADDRESS = "1crypto24HCr17BiMcKd5iVi5D4rsg1nK"

# ---------------- Bitcoin helpers ----------------

def hash160(b):
    return hashlib.new('ripemd160', hashlib.sha256(b).digest()).digest()

def pubkey_to_address(pubkey_bytes):
    vh160 = b'\x00' + hash160(pubkey_bytes)
    checksum = hashlib.sha256(hashlib.sha256(vh160).digest()).digest()[:4]
    return base58.b58encode(vh160 + checksum).decode()

def check_private_key(hex_key):
    if len(hex_key) != 64:
        raise RuntimeError("Private key is not 256 bits")

    priv = bytes.fromhex(hex_key)
    sk = SigningKey.from_string(priv, curve=SECP256k1)
    vk = sk.verifying_key

    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()

    pub_uncompressed = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
    addr_uncompressed = pubkey_to_address(pub_uncompressed)

    prefix = b'\x02' if y % 2 == 0 else b'\x03'
    pub_compressed = prefix + x.to_bytes(32, 'big')
    addr_compressed = pubkey_to_address(pub_compressed)

    print("Uncompressed address:", addr_uncompressed)
    print("Compressed address  :", addr_compressed)

    if TARGET_ADDRESS in (addr_uncompressed, addr_compressed):
        print("\n🔥🔥 MATCH FOUND — PUZZLE SOLVED 🔥🔥")
    else:
        print("\n❌ No match")

# ---------------- Image decoding ----------------

def extract_bars(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    h, w = bw.shape
    bars = []

    for x in range(w):
        column = bw[:, x]
        ys = np.where(column > 0)[0]
        if len(ys) > 5:
            bars.append((x, ys.max() - ys.min()))

    return bars

def estimate_decay(bars):
    xs = np.array([b[0] for b in bars])
    hs = np.array([b[1] for b in bars])

    hs = np.clip(hs, 1, None)
    logh = np.log(hs)

    A = np.vstack([xs, np.ones(len(xs))]).T
    k, _ = np.linalg.lstsq(A, logh, rcond=None)[0]
    return k

def normalize_heights(bars, k):
    normalized = []
    for x, h in bars:
        corrected = h * math.exp(-k * x)
        normalized.append(corrected)
    return np.array(normalized)

def heights_to_bits(norm_heights):
    median = np.median(norm_heights)
    bits = ['1' if h > median else '0' for h in norm_heights]
    return bits

# ---------------- Main ----------------

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Could not load image")

    bars = extract_bars(img)
    print("Detected bars:", len(bars))

    if len(bars) < 256:
        raise RuntimeError("Not enough bars detected")

    k = estimate_decay(bars)
    print("Estimated decay k =", k)

    norm = normalize_heights(bars, k)
    bits = heights_to_bits(norm)

    # Collapse runs (important!)
    collapsed = []
    last = None
    for b in bits:
        if b != last:
            collapsed.append(b)
            last = b

    print("Collapsed bits:", len(collapsed))

    if len(collapsed) != 256:
        raise RuntimeError("Decoded bits != 256 (got %d)" % len(collapsed))

    bitstring = ''.join(collapsed)
    hex_key = hex(int(bitstring, 2))[2:].zfill(64)

    print("\nPrivate key (hex):", hex_key)
    check_private_key(hex_key)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halving_decode_and_check.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])
