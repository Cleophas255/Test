# zden_waveform_solver.py
# Pure frequency-band waveform decoder (no rectangles, no amplitude encoding)

import cv2
import numpy as np
from statistics import median

# -------------------------------
# CONFIG
# -------------------------------

IMAGE_PATH = "zden_waveform.png"
BIN_THRESHOLD = 200
SMOOTH_WINDOW = 9
BAND_EPS = 0.15
FLAT_TAIL_STD = 0.5
FLAT_TAIL_LEN = 30

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# -------------------------------
# WAVEFORM EXTRACTION
# -------------------------------

def extract_waveform_y(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    h, w = img.shape
    _, bw = cv2.threshold(img, BIN_THRESHOLD, 255, cv2.THRESH_BINARY)

    y = np.zeros(w, dtype=np.float32)

    for x in range(w):
        ys = np.where(bw[:, x] > 0)[0]
        if len(ys) == 0:
            y[x] = 0.0
        else:
            y[x] = np.mean(ys) - h / 2

    return y

# -------------------------------
# ZERO CROSSINGS
# -------------------------------

def zero_crossings(y):
    zc = []
    for i in range(1, len(y)):
        if y[i] == 0 or y[i - 1] == 0:
            continue
        if np.sign(y[i]) != np.sign(y[i - 1]):
            zc.append(i)
    return zc

def half_periods(zc):
    return [zc[i] - zc[i - 1] for i in range(1, len(zc))]

# -------------------------------
# SIGNAL CLEANUP
# -------------------------------

def median_smooth(data, window):
    smoothed = []
    half = window // 2
    for i in range(len(data)):
        a = max(0, i - half)
        b = min(len(data), i + half + 1)
        smoothed.append(median(data[a:b]))
    return smoothed

def trim_flat_tail(dx, std_threshold, tail_len):
    if len(dx) < tail_len:
        return dx
    tail = dx[-tail_len:]
    if np.std(tail) < std_threshold:
        return dx[:-tail_len]
    return dx

# -------------------------------
# BAND DETECTION
# -------------------------------

def detect_frequency_bands(dx_smooth, eps):
    freq = [1.0 / d for d in dx_smooth if d > 0]

    bands = []
    current = [freq[0]]

    for i in range(1, len(freq)):
        rel = abs(freq[i] - freq[i - 1]) / freq[i - 1]
        if rel < eps:
            current.append(freq[i])
        else:
            bands.append(current)
            current = [freq[i]]

    bands.append(current)
    return bands

def bands_to_symbols(bands):
    band_values = [median(b) for b in bands]
    uniq = sorted(set(band_values))
    symbols = [uniq.index(v) for v in band_values]
    return symbols, uniq

# -------------------------------
# BASE58 MAPPING
# -------------------------------

def symbols_to_base58(symbols):
    return "".join(
        BASE58_ALPHABET[s] for s in symbols if s < len(BASE58_ALPHABET)
    )

# -------------------------------
# FULL PIPELINE
# -------------------------------

def decode_waveform(image_path):
    y = extract_waveform_y(image_path)

    zc = zero_crossings(y)
    if len(zc) < 10:
        raise RuntimeError("Not enough zero crossings detected")

    dx = half_periods(zc)
    dx = trim_flat_tail(dx, FLAT_TAIL_STD, FLAT_TAIL_LEN)

    dx_smooth = median_smooth(dx, SMOOTH_WINDOW)

    bands = detect_frequency_bands(dx_smooth, BAND_EPS)
    symbols, alphabet = bands_to_symbols(bands)

    return symbols, alphabet

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    symbols, alphabet = decode_waveform(IMAGE_PATH)

    print("Decoded symbol indices:")
    print(symbols)
    print()
    print("Distinct symbol count:", len(alphabet))

    if 56 <= len(alphabet) <= 60:
        print("\nLikely Base58 output:")
        print(symbols_to_base58(symbols))
    else:
        print("\nAlphabet size does NOT match Base58 exactly")
