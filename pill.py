#!/usr/bin/env python3
"""
Waveform Analyzer for Bitcoin Private Key Puzzle
Analyzes waveform images and extracts potential keys using multiple methods
"""

from PIL import Image
import numpy as np
import sys

def load_and_extract_signal(image_path):
    """Load image and extract waveform signal"""
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    pixels = np.array(img)
    height, width = pixels.shape
    
    print(f"Image size: {width}x{height}")
    
    # Extract signal by finding brightest pixel in each column
    signal = []
    peak_positions = []
    
    for x in range(width):
        column = pixels[:, x]
        max_brightness = np.max(column)
        
        if max_brightness > 100:  # Threshold for detecting waveform
            y_position = np.argmax(column)
            # Convert to amplitude (invert Y since it increases downward)
            amplitude = (height - y_position) / height
            signal.append(amplitude)
            
            if max_brightness > 200:  # Higher threshold for peaks
                peak_positions.append({
                    'x': x,
                    'y': y_position,
                    'amplitude': amplitude,
                    'brightness': max_brightness
                })
        else:
            signal.append(0)
    
    print(f"Extracted signal: {len(signal)} samples")
    return np.array(signal), peak_positions, width, height


def detect_peaks(signal, window_size=5):
    """Detect local maxima (peaks) in the signal"""
    peaks = []
    
    for i in range(window_size, len(signal) - window_size):
        is_peak = True
        current = signal[i]
        
        if current < 0.1:  # Skip low amplitude
            continue
        
        # Check if it's a local maximum
        for j in range(-window_size, window_size + 1):
            if j != 0 and signal[i + j] > current:
                is_peak = False
                break
        
        if is_peak:
            peaks.append({
                'position': i,
                'amplitude': current
            })
    
    # Remove peaks that are too close together
    filtered_peaks = []
    for i, peak in enumerate(peaks):
        if i == 0 or peak['position'] - filtered_peaks[-1]['position'] > 3:
            filtered_peaks.append(peak)
    
    print(f"Detected {len(filtered_peaks)} significant peaks")
    return filtered_peaks


def method1_peak_positions_hex(peaks, width):
    """Method 1: Normalize peak X-positions to 0-255 and convert to hex"""
    if not peaks:
        return "No peaks detected"
    
    positions = [p['position'] for p in peaks]
    normalized = [int((pos / width) * 255) for pos in positions]
    hex_string = ''.join([f"{n:02x}" for n in normalized])
    return hex_string[:64]  # Limit to 64 chars (256 bits)


def method2_peak_amplitudes_hex(peaks):
    """Method 2: Convert peak amplitudes to hex"""
    if not peaks:
        return "No peaks detected"
    
    amplitudes = [int(p['amplitude'] * 255) for p in peaks]
    hex_string = ''.join([f"{a:02x}" for a in amplitudes])
    return hex_string[:64]


def method3_positions_single_hex(peaks, width):
    """Method 3: Peak positions as single hex digits (0-F)"""
    if not peaks:
        return "No peaks detected"
    
    positions = [p['position'] for p in peaks]
    normalized = [int((pos / width) * 15) for pos in positions]
    hex_string = ''.join([f"{n:x}" for n in normalized])
    return hex_string[:64]


def method4_amplitudes_single_hex(peaks):
    """Method 4: Peak amplitudes as single hex digits (0-F)"""
    if not peaks:
        return "No peaks detected"
    
    amplitudes = [int(p['amplitude'] * 15) for p in peaks]
    hex_string = ''.join([f"{a:x}" for a in amplitudes])
    return hex_string[:64]


def method5_peak_spacing(peaks):
    """Method 5: Spacing between consecutive peaks"""
    if len(peaks) < 2:
        return "Insufficient peaks"
    
    spacings = []
    for i in range(1, len(peaks)):
        spacing = peaks[i]['position'] - peaks[i-1]['position']
        # Normalize to 0-15 range
        normalized = min(15, spacing // 5)
        spacings.append(f"{normalized:x}")
    
    return ''.join(spacings)[:64]


def method6_combined_position_amplitude(peaks, width):
    """Method 6: Interleaved position and amplitude"""
    if not peaks:
        return "No peaks detected"
    
    result = []
    for p in peaks:
        pos = int((p['position'] / width) * 15)
        amp = int(p['amplitude'] * 15)
        result.append(f"{pos:x}{amp:x}")
    
    return ''.join(result)[:64]


def method7_binary_threshold(signal):
    """Method 7: Binary encoding based on threshold"""
    threshold = np.mean(signal)
    binary = ''.join(['1' if s > threshold else '0' for s in signal])
    
    # Convert binary to hex (4 bits at a time)
    hex_string = ''
    for i in range(0, len(binary), 4):
        chunk = binary[i:i+4]
        if len(chunk) == 4:
            hex_string += f"{int(chunk, 2):x}"
    
    return hex_string[:64]


def method8_envelope_detection(signal, window_size=10):
    """Method 8: Envelope detection (amplitude tracking)"""
    envelope = []
    
    for i in range(len(signal)):
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size)
        window_max = np.max(signal[start:end])
        envelope.append(window_max)
    
    # Downsample envelope to 64 values
    step = max(1, len(envelope) // 64)
    sampled = [int(envelope[i] * 15) for i in range(0, len(envelope), step)]
    hex_string = ''.join([f"{s:x}" for s in sampled[:64]])
    
    return hex_string


def method9_derivative_analysis(signal):
    """Method 9: First derivative (rate of change)"""
    derivative = np.diff(signal)
    
    # Find zero crossings in derivative (inflection points)
    crossings = []
    for i in range(1, len(derivative)):
        if derivative[i-1] * derivative[i] < 0:
            crossings.append(i)
    
    # Convert crossing positions to hex
    if not crossings:
        return "No crossings detected"
    
    normalized = [int((c / len(signal)) * 255) for c in crossings]
    hex_string = ''.join([f"{n:02x}" for n in normalized])
    return hex_string[:64]


def calculate_statistics(signal):
    """Calculate statistical properties of the signal"""
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    
    # Skewness and kurtosis (simplified)
    centered = signal - mean
    skewness = np.mean(centered**3) / (std**3) if std > 0 else 0
    kurtosis = np.mean(centered**4) / (std**4) if std > 0 else 0
    
    return {
        'mean': mean,
        'std': std,
        'variance': var,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def analyze_waveform(image_path):
    """Main analysis function"""
    print("="*60)
    print("WAVEFORM ANALYZER FOR BITCOIN PRIVATE KEY PUZZLE")
    print("="*60)
    
    # Load and extract signal
    signal, raw_peaks, width, height = load_and_extract_signal(image_path)
    
    # Detect peaks
    peaks = detect_peaks(signal)
    
    # Calculate statistics
    stats = calculate_statistics(signal)
    
    print("\nStatistical Properties:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std Dev: {stats['std']:.4f}")
    print(f"  Skewness: {stats['skewness']:.4f}")
    print(f"  Kurtosis: {stats['kurtosis']:.4f}")
    
    # Generate keys using all methods
    print("\n" + "="*60)
    print("POTENTIAL KEYS (9 METHODS)")
    print("="*60)
    
    methods = [
        ("Method 1: Peak Positions (2 hex)", method1_peak_positions_hex(peaks, width)),
        ("Method 2: Peak Amplitudes (2 hex)", method2_peak_amplitudes_hex(peaks)),
        ("Method 3: Positions (1 hex digit)", method3_positions_single_hex(peaks, width)),
        ("Method 4: Amplitudes (1 hex digit)", method4_amplitudes_single_hex(peaks)),
        ("Method 5: Peak Spacing", method5_peak_spacing(peaks)),
        ("Method 6: Combined Pos+Amp", method6_combined_position_amplitude(peaks, width)),
        ("Method 7: Binary Threshold", method7_binary_threshold(signal)),
        ("Method 8: Envelope Detection", method8_envelope_detection(signal)),
        ("Method 9: Derivative Analysis", method9_derivative_analysis(signal))
    ]
    
    results = {}
    for name, value in methods:
        print(f"\n{name}:")
        print(f"  {value}")
        print(f"  Length: {len(value)} characters")
        results[name] = value
    
    # Show peak details
    print("\n" + "="*60)
    print(f"PEAK DETAILS (showing first 20 of {len(peaks)})")
    print("="*60)
    
    for i, peak in enumerate(peaks[:20]):
        print(f"Peak {i+1}: Position={peak['position']}, Amplitude={peak['amplitude']:.3f}")
    
    return results


def save_results(results, output_file="waveform_analysis.txt"):
    """Save results to a text file"""
    with open(output_file, 'w') as f:
        f.write("WAVEFORM ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for method, value in results.items():
            f.write(f"{method}:\n")
            f.write(f"{value}\n")
            f.write(f"Length: {len(value)} characters\n\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python waveform_analyzer.py <image_path>")
        print("Example: python waveform_analyzer.py waveform.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        results = analyze_waveform(image_path)
        save_results(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Try each hex string as a potential Bitcoin private key")
        print("2. Bitcoin private keys are 64 hex characters (256 bits)")
        print("3. Or in WIF format starting with 5, K, or L")
        
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)