import cv2
import numpy as np
import hashlib
import base58
from ecdsa import SigningKey, SECP256k1
from sklearn.cluster import KMeans
import sys

TARGET_ADDRESS = "1crypto24HCr17BiMcKd5iVi5D4rsg1nK"
EXPECTED_BITS = 256

# ---------------- Bitcoin helpers ----------------

def hash160(b):
    return hashlib.new('ripemd160', hashlib.sha256(b).digest()).digest()

def pubkey_to_address(pubkey_bytes):
    vh160 = b'\x00' + hash160(pubkey_bytes)
    checksum = hashlib.sha256(hashlib.sha256(vh160).digest()).digest()[:4]
    return base58.b58encode(vh160 + checksum).decode()

def check_private_key(bitstring, label=""):
    if len(bitstring) != 256:
        return False

    hexkey = hex(int(bitstring, 2))[2:].zfill(64)
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
        print("Address:", TARGET_ADDRESS)
        sys.exit(0)

    return False

# ---------------- Edge-timing decoder ----------------

def decode_edge_timing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hard threshold (image is clean)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Collapse vertically → 1D signal
    projection = np.sum(bw > 0, axis=0)

    # Normalize
    projection = projection / np.max(projection)

    # Binary signal
    signal = projection > np.mean(projection)

    # Edge detection
    edges = np.diff(signal.astype(int))
    edge_positions = np.where(edges != 0)[0]

    print("Edges detected:", len(edge_positions))

    # Distances between edges
    intervals = np.diff(edge_positions)

    print("Intervals:", len(intervals))

    # Cluster intervals into 2 groups
    data = intervals.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    zero_cluster = np.argmin(centers)
    one_cluster = np.argmax(centers)

    bits = ['1' if l == one_cluster else '0' for l in labels]

    return ''.join(bits)

# ---------------- Main logic ----------------

def main(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Failed to load image")

    bitstream = decode_edge_timing(img)

    print("Raw bits:", len(bitstream))
    print("First 64 bits:", bitstream[:64])

    # Try variants (VERY important)
    variants = {
        "raw": bitstream,
        "inverted": ''.join('1' if b == '0' else '0' for b in bitstream),
        "reversed": bitstream[::-1],
        "reversed+inverted": ''.join('1' if b == '0' else '0' for b in bitstream[::-1]),
    }

    for name, bits in variants.items():
        if len(bits) >= EXPECTED_BITS:
            trimmed = bits[:EXPECTED_BITS]
            check_private_key(trimmed, name)

    print("❌ No match found yet")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halving_edge_timing.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])
