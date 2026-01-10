import cv2
import numpy as np
import sys
from scipy.signal import find_peaks

# ---------- CONFIG ----------
EXPECTED_BITS = 256
MIN_PEAK_DISTANCE = 3  # pixels between bars (critical)
THRESHOLD = 180
# ----------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Image not found")
    return img

def preprocess(img):
    _, bw = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)
    bw = cv2.medianBlur(bw, 3)
    return bw

def find_bar_positions(bw):
    # vertical projection
    projection = np.sum(bw > 0, axis=0)

    # normalize
    projection = projection / projection.max()

    # find peaks (each peak ≈ one bar)
    peaks, _ = find_peaks(
        projection,
        height=0.2,
        distance=MIN_PEAK_DISTANCE
    )

    return peaks, projection

def measure_bar_heights(bw, peaks):
    h, _ = bw.shape
    heights = []

    for x in peaks:
        column = bw[:, x]
        ys = np.where(column > 0)[0]
        if len(ys) == 0:
            heights.append(0)
        else:
            heights.append(ys.max() - ys.min())

    return np.array(heights, dtype=float)

def estimate_decay(heights):
    x = np.arange(len(heights))
    nonzero = heights > 0
    x = x[nonzero]
    y = np.log(heights[nonzero])

    k, _ = np.polyfit(x, y, 1)
    return k

def reverse_decay(heights, k):
    x = np.arange(len(heights))
    return heights * np.exp(-k * x)

def heights_to_bits(norm_heights):
    # k-means (2 clusters)
    data = norm_heights.reshape(-1, 1).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        data, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    # higher = 1
    bit_map = labels.flatten()
    centers = centers.flatten()
    one_cluster = np.argmax(centers)

    bits = ''.join('1' if b == one_cluster else '0' for b in bit_map)
    return bits

def main(path):
    img = load_image(path)
    bw = preprocess(img)

    peaks, projection = find_bar_positions(bw)
    print("Detected bars:", len(peaks))

    if len(peaks) < EXPECTED_BITS * 0.9:
        print("⚠️ Warning: fewer bars than expected")

    heights = measure_bar_heights(bw, peaks)

    k = estimate_decay(heights)
    print("Estimated decay k =", k)

    norm_heights = reverse_decay(heights, k)

    bits = heights_to_bits(norm_heights)

    print("Total bits:", len(bits))
    print("First 64 bits:", bits[:64])

    if len(bits) < EXPECTED_BITS:
        raise RuntimeError("Not enough bits extracted")

    print("Binary extraction complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python halv_decoder.py cryptoHALV.png")
        sys.exit(1)

    main(sys.argv[1])