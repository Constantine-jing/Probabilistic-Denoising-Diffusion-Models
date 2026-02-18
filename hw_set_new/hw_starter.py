"""
STAT 9100 — Homework: Exploring the DDPM Forward Process
=========================================================
Starter code — fill in the ### TODO ### sections.

Requirements: numpy, matplotlib, Pillow (PIL)
    pip install numpy matplotlib Pillow

Run:  python hw_starter.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════
# HELPER: Create a test image (colored checkerboard)
# ════════════════════════════════════════════════════════════

def make_test_image(size=128):
    """Creates a colorful 8x8 checkerboard image, shape (size, size, 3)."""
    block = size // 8
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.9, 0.3, 0.2]
            else:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.2, 0.5, 0.9]
    return img


# ════════════════════════════════════════════════════════════
# HELPER: Forward process (noise an image)
# ════════════════════════════════════════════════════════════

def q_sample(x0, t, alpha_bar):
    """
    Forward process: sample x_t given x_0.
    
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * eps
    
    Args:
        x0:        clean image, shape (H, W, 3), values in [0, 1]
        t:         integer timestep (0-indexed, so t=0 means alpha_bar[0])
        alpha_bar: 1D array of cumulative products, length T
    
    Returns:
        noisy image x_t, clipped to [0, 1] for display
    """
    eps = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    return xt.clip(0, 1)


# ════════════════════════════════════════════════════════════
# SCHEDULE FUNCTIONS
# ════════════════════════════════════════════════════════════

def linear_schedule(T=1000):
    """
    Linear beta schedule from Ho et al. (2020).
    beta goes from 1e-4 to 2e-2 linearly.
    
    Returns:
        alpha_bar: 1D array of length T
    """
    betas = np.linspace(1e-4, 2e-2, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar


def cosine_schedule(T=1000):
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    
    Returns:
        alpha_bar: 1D array of length T
    """
    s = 0.008
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    ab = ab[1:]  # length T, for t = 1..T
    # derive betas and clip
    betas = 1 - ab / np.concatenate([[1.0], ab[:-1]])
    betas = np.clip(betas, 0, 0.999)
    alpha_bar = np.cumprod(1 - betas)
    return alpha_bar


def sqrt_schedule(T=1000):
    """
    Square-root schedule: alpha_bar_t = 1 - sqrt(t / T)
    
    ### TODO (Question 1a) ###
    Implement this function.
    
    Hints:
    - Create an array t_vals = np.arange(1, T+1) so t goes from 1 to T
    - Compute alpha_bar = 1 - np.sqrt(t_vals / T)
    - Clip alpha_bar to be >= 1e-6 (to avoid numerical issues)
    
    Returns:
        alpha_bar: 1D array of length T
    """
    # ---- YOUR CODE HERE ----

    pass  # Remove this line after implementing

    # ---- END YOUR CODE ----


# ════════════════════════════════════════════════════════════
# HELPER: Compute SNR from alpha_bar
# ════════════════════════════════════════════════════════════

def compute_snr(alpha_bar):
    """SNR(t) = alpha_bar_t / (1 - alpha_bar_t)"""
    return alpha_bar / (1 - alpha_bar + 1e-12)


# ════════════════════════════════════════════════════════════
# QUESTION 1b: Plot SNR for all three schedules
# ════════════════════════════════════════════════════════════

def plot_snr_comparison():
    """Plot SNR(t) for linear, cosine, and sqrt schedules."""
    T = 1000
    ab_lin = linear_schedule(T)
    ab_cos = cosine_schedule(T)
    ab_sqrt = sqrt_schedule(T)
    
    if ab_sqrt is None:
        print("ERROR: sqrt_schedule() returned None. Implement it first!")
        return
    
    snr_lin = compute_snr(ab_lin)
    snr_cos = compute_snr(ab_cos)
    snr_sqrt = compute_snr(ab_sqrt)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: alpha_bar curves
    axes[0].plot(ab_lin, label="Linear", linewidth=2, color="#ef4444")
    axes[0].plot(ab_cos, label="Cosine", linewidth=2, color="#3b82f6")
    axes[0].plot(ab_sqrt, label="Sqrt (yours)", linewidth=2, color="#10b981")
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel("ᾱ_t (signal fraction)")
    axes[0].set_title("Alpha-bar comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: SNR curves (log scale)
    axes[1].plot(snr_lin, label="Linear", linewidth=2, color="#ef4444")
    axes[1].plot(snr_cos, label="Cosine", linewidth=2, color="#3b82f6")
    axes[1].plot(snr_sqrt, label="Sqrt (yours)", linewidth=2, color="#10b981")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("SNR(t)  [log scale]")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("q1_snr_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: q1_snr_comparison.png")


# ════════════════════════════════════════════════════════════
# QUESTION 2a: Visual comparison — linear vs sqrt
# ════════════════════════════════════════════════════════════

def plot_schedule_visual_comparison():
    """
    Generate a 2-row figure:
      Top row:    forward process with LINEAR schedule
      Bottom row: forward process with SQRT schedule
    at timesteps [0, 200, 400, 600, 800, 999].
    Use the same random seed for both rows.
    """
    T = 1000
    x0 = make_test_image()
    ab_lin = linear_schedule(T)
    ab_sqrt = sqrt_schedule(T)
    
    if ab_sqrt is None:
        print("ERROR: sqrt_schedule() returned None. Implement it first!")
        return
    
    timesteps = [0, 200, 400, 600, 800, 999]
    
    ### TODO (Question 2a) ###
    # Create a figure with 2 rows and len(timesteps) columns.
    # For each timestep t:
    #   - Set np.random.seed(t) before EACH call to q_sample
    #     (so both rows use the same noise)
    #   - Top row: q_sample(x0, t, ab_lin)
    #   - Bottom row: q_sample(x0, t, ab_sqrt)
    #   - Show image with ax.imshow(), add title f"t={t}", turn off axis
    # Add row labels: "Linear" and "Sqrt"
    # Save as "q2_visual_comparison.png"
    
    # ---- YOUR CODE HERE ----
    
    
    pass  # Remove this line after implementing
    
    
    # ---- END YOUR CODE ----


# ════════════════════════════════════════════════════════════
# QUESTION 3a: Effect of T on cosine schedule
# ════════════════════════════════════════════════════════════

def plot_T_comparison():
    """
    Plot SNR curves for cosine schedule with T=50, T=200, T=1000.
    X-axis should be the fraction t/T (0 to 1), not the raw timestep.
    This lets us compare the shape regardless of T.
    """
    T_values = [50, 200, 1000]
    colors = ["#f59e0b", "#8b5cf6", "#3b82f6"]
    
    ### TODO (Question 3a) ###
    # For each T in T_values:
    #   1. Compute alpha_bar using cosine_schedule(T)
    #   2. Compute snr using compute_snr(alpha_bar)
    #   3. Create the x-axis as fractions: np.arange(1, T+1) / T
    #   4. Plot snr vs fraction on the SAME axes (use log scale for y)
    # Add xlabel "t / T", ylabel "SNR (log scale)", legend, grid
    # Save as "q3_T_comparison.png"
    
    # ---- YOUR CODE HERE ----
    
    
    pass  # Remove this line after implementing
    
    
    # ---- END YOUR CODE ----


# ════════════════════════════════════════════════════════════
# MAIN — Run all questions
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("STAT 9100 Homework — DDPM Forward Process")
    print("=" * 60)
    
    print("\n📊 Question 1: SNR Comparison (3 schedules)")
    plot_snr_comparison()
    
    print("\n🖼️  Question 2: Visual Comparison (linear vs sqrt)")
    plot_schedule_visual_comparison()
    
    print("\n📈 Question 3: Effect of T on cosine schedule")
    plot_T_comparison()
    
    print("\n✏️  Question 4: Written answers — see hw_questions.md")
    print("\nDone! Check the generated .png files.")
