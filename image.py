import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import ecdsa
import base58

# -----------------------------
# 1. Load image
# -----------------------------
img = cv2.imread("cryptoHALV.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("Image not found: cryptoHALV.png")

h, w = img.shape
print(f"[+] Image loaded: {w} x {h}")

# -----------------------------
# 2. Threshold to isolate waveform
# -----------------------------
_, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# -----------------------------
# 3. Extract 1D signal
# -----------------------------
signal = np.zeros(w)

for x in range(w):
    ys = np.where(bw[:, x] > 0)[0]
    if len(ys) > 0:
        signal[x] = np.mean(ys)
    else:
        signal[x] = np.nan

signal = np.nan_to_num(signal)

# invert & normalize
signal = -(signal - np.mean(signal))
maxv = np.max(np.abs(signal))
if maxv != 0:
    signal = signal / maxv

print("[+] Signal extracted")

# -----------------------------
# 4. Plot signal
# -----------------------------
plt.figure(figsize=(14, 4))
plt.plot(signal, linewidth=0.6)
plt.title("Recovered 1D Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Histogram (quantization test)
# -----------------------------
plt.figure()
plt.hist(signal, bins=200)
plt.title("Amplitude Distribution")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Trim low-amplitude tail
# -----------------------------
usable = signal[np.abs(signal) > 0.05]
print(f"[+] Usable samples: {len(usable)}")

if len(usable) < 256:
    raise RuntimeError("Not enough usable samples to extract 256 bits")

# -----------------------------
# 7. Downsample to 256 points
# -----------------------------
N = 256
idx = np.linspace(0, len(usable) - 1, N).astype(int)
samples = usable[idx]

plt.figure(figsize=(14, 4))
plt.plot(samples, marker="o")
plt.title("256 Sample Points")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Convert samples to bits
# -----------------------------
bits = (samples > 0).astype(int)
bitstring = "".join(bits.astype(str))

print("[+] Bitstring length:", len(bitstring))
print("[+] Bitstring (first 64 bits):", bitstring[:64])

# -----------------------------
# 9. Bits → private key → BTC address
# -----------------------------
def privkey_to_address(privkey_bytes):
    sk = ecdsa.SigningKey.from_string(privkey_bytes, curve=ecdsa.SECP256k1)
    vk = sk.verifying_key
    pub = b"\x04" + vk.to_string()
    h160 = hashlib.new("ripemd160", hashlib.sha256(pub).digest()).digest()
    addr = base58.b58encode_check(b"\x00" + h160)
    return addr.decode()

priv_int = int(bitstring, 2)
priv_bytes = priv_int.to_bytes(32, byteorder="big")

address = privkey_to_address(priv_bytes)

print("\n================ RESULT ================")
print("Derived BTC address:")
print(address)
print("=======================================\n")