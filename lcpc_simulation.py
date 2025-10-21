import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc  # For theoretical uncoded BER

# Define the matrices for LCPC (9,4) as per the paper
G = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 1, 1]
], dtype=int)

H = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1]
], dtype=int)

k = 4  # Information bits
n = 9  # Codeword length
rate = k / n

# Generate lookup table for syndromes (up to 2 errors, as per paper)
def generate_lookup_table(H, max_errors=2):
    lookup = {}
    # Zero error
    ep = np.zeros(n, dtype=int)
    sy = (H @ ep) % 2
    lookup[tuple(sy)] = ep.copy()

    # Single errors
    for i in range(n):
        ep = np.zeros(n, dtype=int)
        ep[i] = 1
        sy = (H @ ep) % 2
        sy_tuple = tuple(sy)
        if sy_tuple in lookup:
            print(f"Warning: Syndrome collision for single error at position {i}")
        lookup[sy_tuple] = ep.copy()

    # Double errors
    for i in range(n):
        for j in range(i + 1, n):
            ep = np.zeros(n, dtype=int)
            ep[i] = 1
            ep[j] = 1
            sy = (H @ ep) % 2
            sy_tuple = tuple(sy)
            if sy_tuple in lookup:
                print(f"Warning: Syndrome collision for double error at positions {i},{j}")
            lookup[sy_tuple] = ep.copy()

    print(f"Lookup table generated with {len(lookup)} entries")
    return lookup

lookup = generate_lookup_table(H)

# Encoding function
def encode(u, G):
    return (u @ G) % 2

# Decoding function
def decode(r, H, lookup, k):
    sy = (H @ r) % 2
    sy_tuple = tuple(sy)
    if sy_tuple in lookup:
        ep = lookup[sy_tuple]
        corrected = (r - ep) % 2
        return corrected[:k]  # Extract information bits
    else:
        # Unc correctable: detect error, but return received info bits for BER calc
        return r[:k]

# BPSK modulation (0 -> 1, 1 -> -1)
def bpsk_mod(c):
    return 1 - 2 * c

# AWGN channel
def awgn_channel(x, snr_db, rate):
    ebno = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * rate * ebno))
    noise = sigma * np.random.randn(len(x))
    return x + noise

# Hard decision
def hard_decide(y):
    return (y < 0).astype(int)

# Simulation function
def simulate_ber(snr_range, num_simulations=10000):
    bers = []
    for snr_db in snr_range:
        bit_errors = 0
        total_bits = 0
        for _ in range(num_simulations):
            # Generate random message
            u = np.random.randint(0, 2, k)
            # Encode
            c = encode(u, G)
            # Modulate
            x = bpsk_mod(c)
            # Channel
            y = awgn_channel(x, snr_db, rate)
            # Demodulate/Hard decide
            r = hard_decide(y)
            # Decode
            u_hat = decode(r, H, lookup, k)
            # Count errors
            bit_errors += np.sum(u != u_hat)
            total_bits += k
        ber = bit_errors / total_bits if total_bits > 0 else 0
        bers.append(ber)
        print(f"SNR: {snr_db} dB, BER: {ber}")
    return bers

# Theoretical uncoded BPSK BER for comparison
def theoretical_uncoded_ber(snr_range):
    return 0.5 * erfc(np.sqrt(10 ** (snr_range / 10)))

# Run simulation
snr_range = np.arange(0, 11, 1)  # SNR from 0 to 10 dB
bers_lcpc = simulate_ber(snr_range)

# Uncoded theoretical
bers_uncoded = theoretical_uncoded_ber(snr_range)

# Plot results
plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, bers_uncoded, 'r--', label='Uncoded BPSK (Theoretical)')
plt.semilogy(snr_range, bers_lcpc, 'b-o', label='LCPC (9,4) Simulated')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. SNR for LCPC Code over AWGN Channel')
plt.grid(True, which='both')
plt.legend()
plt.show()

# Save plot for your report
plt.savefig('lcpc_ber_plot.png')
