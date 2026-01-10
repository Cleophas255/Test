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

# ---------------- Image decoding ----------------

def extract_amplitudes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    h, w = bw.shape
    slice_w = w // BITS

    amplitudes = []

    for i in range(BITS):
        x0 = i * slice_w
        x1 = min(w, (i + 1) * slice_w)
        col = bw[:, x0:x1]

        ys = np.where(col > 0)[0]
        if len(ys) == 0:
            amplitudes.append(0)
            continue

        baseline = np.median(ys)
        peak = ys.min()   # waveform goes upward
        amplitudes.append(abs(baseline - peak))

    return np.array(amplitudes, dtype=np.float64)

def normalize_decay(amplitudes):
    xs = np.arange(len(amplitudes))
    amps = np.clip(amplitudes, 1, None)

    loga = np.log(amps)
    k, c = np.polyfit(xs, loga, 1)

    # normalize relative to first bit
    norm = amps * np.exp(-k * xs)
    norm /= norm[0]

    return norm

def bits_from_kmeans(values):
    v = values.reshape(-1, 1).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        v, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01),
        10, cv2.KMEANS_PP_CENTERS
    )

    hi = np.argmax(centers)
    return ['1' if l == hi else '0' for l in labels.flatten()]

# ---------------- Main ----------------

def main(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Image load failed")

    amps = extract_amplitudes(img)
    print("Samples:", len(amps))

    norm = normalize_decay(amps)

    bits = bits_from_kmeans(norm)
    bitstring = ''.join(bits)

    print("First 64 bits:", bitstring[:64])

    hexkey = hex(int(bitstring, 2))[2:].zfill(64)
    print("\nPrivate key:", hexkey)

    check_key(hexkey)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halving_decode.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])
