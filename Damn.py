import cv2
import numpy as np
import math
from hashlib import sha256, new as new_hash
import base58

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = "cryptoHALV.png"
WHITE_THRESHOLD = 200   # grayscale threshold for white lines

# -------------------------------
# LOAD IMAGE
# -------------------------------
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("Failed to load image")

h, w = img.shape
print(f"Image size: {w} x {h}")

# -------------------------------
# BINARIZE (white lines only)
# -------------------------------
_, bw = cv2.threshold(img, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

# -------------------------------
# COLUMN HEIGHT EXTRACTION
# -------------------------------
column_heights = np.zeros(w, dtype=int)

for x in range(w):
    ys = np.where(bw[:, x] > 0)[0]
    if len(ys) > 0:
        column_heights[x] = ys[-1] - ys[0]
    else:
        column_heights[x] = 0

# -------------------------------
# COLLAPSE COLUMNS → BARS
# -------------------------------
bars = []
current = []

for hgt in column_heights:
    if hgt > 0:
        current.append(hgt)
    else:
        if current:
            bars.append(max(current))
            current = []

if current:
    bars.append(max(current))

bars = np.array(bars)
print("Collapsed bars:", len(bars))

# -------------------------------
# ESTIMATE EXPONENTIAL DECAY
# -------------------------------
x = np.arange(len(bars))
valid = bars > np.percentile(bars, 60)  # assume top bars are 1s

log_heights = np.log(bars[valid])
x_valid = x[valid]

k, b = np.polyfit(x_valid, log_heights, 1)
k = -k  # decay constant
print("Estimated decay k =", k)

# -------------------------------
# REVERSE HALVING
# -------------------------------
normalized = bars * np.exp(k * x)

# -------------------------------
# BINARIZE BARS
# -------------------------------
threshold = (normalized.max() + normalized.min()) / 2
bits = (normalized > threshold).astype(int)

print("Total bits:", len(bits))
print("First 64 bits:")
print("".join(map(str, bits[:64])))

# -------------------------------
# FORCE EXACT 256 BITS
# -------------------------------
if len(bits) < 256:
    raise RuntimeError("Not enough bits extracted")

bits = bits[:256]

# -------------------------------
# BINARY → HEX PRIVATE KEY
# -------------------------------
bin_str = "".join(map(str, bits))
priv_key_hex = hex(int(bin_str, 2))[2:].zfill(64)
print("\nPrivate key (hex):")
print(priv_key_hex)

# -------------------------------
# PRIVATE KEY → BITCOIN ADDRESS
# -------------------------------
def private_key_to_address(priv_hex):
    priv_bytes = bytes.fromhex(priv_hex)

    sha = sha256(priv_bytes).digest()
    ripemd = new_hash('ripemd160', sha).digest()

    versioned = b'\x00' + ripemd
    checksum = sha256(sha256(versioned).digest()).digest()[:4]
    address = base58.b58encode(versioned + checksum).decode()

    return address

address = private_key_to_address(priv_key_hex)
print("\nBitcoin address:")
print(address)