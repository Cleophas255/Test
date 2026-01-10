import cv2
import numpy as np
import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
from sklearn.cluster import KMeans
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

def check_bits(bits, label):
    if len(bits) < BITS:
        return False

    bits = bits[:BITS]
    hexkey = hex(int(bits, 2))[2:].zfill(64)
    priv = bytes.fromhex(hexkey)

    sk = SigningKey.from_string(priv, curve=SECP256k1)
    vk = sk.verifying_key

    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()

    pub_u = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
    pub_c = (b'\x02' if y % 2 == 0 else b'\x03') + x.to_bytes(32, 'big')

    a1 = pubkey_to_address(pub_u)
    a2 = pubkey_to_address(pub_c)

    if TARGET_ADDRESS in (a1, a2):
        print("\n🔥 SOLVED 🔥")
        print("Variant:", label)
        print("Private key:", hexkey)
        sys.exit(0)

    return False

# ---------------- Run-length decoder ----------------

def extract_signal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    proj = np.sum(bw > 0, axis=0)
    proj = proj / np.max(proj)

    return proj > np.mean(proj)

def run_lengths(signal):
    runs = []
    current = signal[0]
    length = 1

    for s in signal[1:]:
        if s == current:
            length += 1
        else:
            runs.append((current, length))
            current = s
            length = 1

    runs.append((current, length))
    return runs

def cluster_lengths(runs, state):
    lengths = np.array([l for s, l in runs if s == state]).reshape(-1, 1)

    if len(lengths) < 10:
        return None

    kmeans = KMeans(n_clusters=2, n_init=20).fit(lengths)
    centers = kmeans.cluster_centers_.flatten()
    short = np.argmin(centers)
    long = np.argmax(centers)

    return lengths.flatten(), kmeans.labels_, short, long

# ---------------- Main ----------------

def main(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Image load failed")

    signal = extract_signal(img)
    runs = run_lengths(signal)

    print("Total runs:", len(runs))

    for state_name, state in [("ON", True), ("OFF", False)]:
        clustered = cluster_lengths(runs, state)
        if not clustered:
            continue

        lengths, labels, short, long = clustered
        bits = []

        idx = 0
        for s, l in runs:
            if s == state:
                bits.append('1' if labels[idx] == long else '0')
                idx += 1

        bitstring = ''.join(bits)
        print(f"{state_name} bits:", len(bitstring))

        # Try variants
        variants = {
            f"{state_name}_raw": bitstring,
            f"{state_name}_inv": ''.join('1' if b == '0' else '0' for b in bitstring),
            f"{state_name}_rev": bitstring[::-1],
            f"{state_name}_rev_inv": ''.join('1' if b == '0' else '0' for b in bitstring[::-1]),
        }

        for name, b in variants.items():
            check_bits(b, name)

    print("❌ No solution yet")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halving_runlength.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])
