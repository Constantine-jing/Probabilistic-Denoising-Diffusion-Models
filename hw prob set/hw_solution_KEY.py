"""
STAT 9100 — Homework SOLUTION KEY
===================================
DO NOT distribute to students.

This file contains all completed TODO sections and sample written answers.
"""

import numpy as np
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════
# HELPERS (same as starter)
# ════════════════════════════════════════════════════════════

def make_test_image(size=128):
    block = size // 8
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.9, 0.3, 0.2]
            else:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.2, 0.5, 0.9]
    return img


def q_sample(x0, t, alpha_bar):
    eps = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    return xt.clip(0, 1)


def linear_schedule(T=1000):
    betas = np.linspace(1e-4, 2e-2, T, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def cosine_schedule(T=1000):
    s = 0.008
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    ab = ab[1:]
    betas = 1 - ab / np.concatenate([[1.0], ab[:-1]])
    betas = np.clip(betas, 0, 0.999)
    return np.cumprod(1 - betas)


def compute_snr(alpha_bar):
    return alpha_bar / (1 - alpha_bar + 1e-12)


# ════════════════════════════════════════════════════════════
# QUESTION 1a SOLUTION: sqrt schedule
# ════════════════════════════════════════════════════════════

def sqrt_schedule(T=1000):
    """
    SOLUTION: alpha_bar_t = 1 - sqrt(t / T)
    """
    t_vals = np.arange(1, T + 1, dtype=np.float64)
    alpha_bar = 1.0 - np.sqrt(t_vals / T)
    alpha_bar = np.clip(alpha_bar, 1e-6, 1.0)  # avoid numerical issues
    return alpha_bar


# ════════════════════════════════════════════════════════════
# QUESTION 1b SOLUTION: SNR comparison plot
# ════════════════════════════════════════════════════════════

def plot_snr_comparison():
    T = 1000
    ab_lin = linear_schedule(T)
    ab_cos = cosine_schedule(T)
    ab_sqrt = sqrt_schedule(T)

    snr_lin = compute_snr(ab_lin)
    snr_cos = compute_snr(ab_cos)
    snr_sqrt = compute_snr(ab_sqrt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ab_lin, label="Linear", linewidth=2, color="#ef4444")
    axes[0].plot(ab_cos, label="Cosine", linewidth=2, color="#3b82f6")
    axes[0].plot(ab_sqrt, label="Sqrt", linewidth=2, color="#10b981")
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel("ᾱ_t (signal fraction)")
    axes[0].set_title("Alpha-bar comparison")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(snr_lin, label="Linear", linewidth=2, color="#ef4444")
    axes[1].plot(snr_cos, label="Cosine", linewidth=2, color="#3b82f6")
    axes[1].plot(snr_sqrt, label="Sqrt", linewidth=2, color="#10b981")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Timestep t")
    axes[1].set_ylabel("SNR(t)  [log scale]")
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("q1_snr_comparison_SOLUTION.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: q1_snr_comparison_SOLUTION.png")


"""
QUESTION 1b WRITTEN ANSWER:

The sqrt schedule falls between linear and cosine in terms of how quickly
it destroys information. It drops faster than cosine at early timesteps
(the signal decays more aggressively at first), but slower than linear at
later timesteps (it doesn't reach near-zero as early). Overall, the sqrt
schedule is more aggressive than cosine early on, but still retains some
useful signal in the later timesteps, unlike linear which essentially
wastes the last ~25% of the process on pure noise.
"""


# ════════════════════════════════════════════════════════════
# QUESTION 2a SOLUTION: Visual comparison
# ════════════════════════════════════════════════════════════

def plot_schedule_visual_comparison():
    T = 1000
    x0 = make_test_image()
    ab_lin = linear_schedule(T)
    ab_sqrt = sqrt_schedule(T)

    timesteps = [0, 200, 400, 600, 800, 999]

    # SOLUTION
    fig, axes = plt.subplots(2, len(timesteps), figsize=(16, 5))

    for col, t in enumerate(timesteps):
        # Same seed for both rows — fair comparison
        np.random.seed(t)
        img_lin = q_sample(x0, t, ab_lin)
        np.random.seed(t)
        img_sqrt = q_sample(x0, t, ab_sqrt)

        axes[0, col].imshow(img_lin)
        axes[0, col].set_title(f"t={t}", fontsize=11, fontweight='bold')
        axes[0, col].axis("off")

        axes[1, col].imshow(img_sqrt)
        axes[1, col].set_title(f"t={t}", fontsize=11)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Linear", fontsize=12, fontweight='bold',
                           rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("Sqrt", fontsize=12, fontweight='bold',
                           rotation=0, labelpad=50)
    fig.suptitle("Forward process: Linear vs Sqrt schedule (same noise)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("q2_visual_comparison_SOLUTION.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: q2_visual_comparison_SOLUTION.png")


"""
QUESTION 2b WRITTEN ANSWER:

With the linear schedule, the image becomes essentially unrecognizable
around t=400-500. With the sqrt schedule, the image holds on a bit longer
— it stays somewhat recognizable until around t=500-600, because the sqrt
alpha_bar curve decays more slowly in the middle range.

The linear schedule wastes more timesteps on pure noise: after about
t=750, both the linear and sqrt images are indistinguishable from random
noise, but linear reaches this point sooner, meaning roughly the last 25%
of linear timesteps contribute no useful denoising signal.
"""


# ════════════════════════════════════════════════════════════
# QUESTION 3a SOLUTION: Effect of T
# ════════════════════════════════════════════════════════════

def plot_T_comparison():
    T_values = [50, 200, 1000]
    colors = ["#f59e0b", "#8b5cf6", "#3b82f6"]

    # SOLUTION
    plt.figure(figsize=(8, 5))

    for T, color in zip(T_values, colors):
        ab = cosine_schedule(T)
        snr = compute_snr(ab)
        fractions = np.arange(1, T + 1) / T  # x-axis: t/T from 0 to 1
        plt.plot(fractions, snr, label=f"T={T}", linewidth=2, color=color)

    plt.yscale("log")
    plt.xlabel("t / T  (fraction of total steps)")
    plt.ylabel("SNR(t)  [log scale]")
    plt.title("Cosine schedule SNR at different T values")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("q3_T_comparison_SOLUTION.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: q3_T_comparison_SOLUTION.png")


"""
QUESTION 3b WRITTEN ANSWER:

Yes, the cosine schedule maintains a very similar shape regardless of T
when plotted against the fraction t/T. The three curves (T=50, 200, 1000)
nearly overlap. This scale-invariance is important for fast sampling
because it means you can train a model with T=1000 but sample with far
fewer steps (e.g., T=100) by evenly spacing timesteps, and the noise
levels at those steps will still cover the same range in a smooth way.
This is exactly what Nichol & Dhariwal exploit: they train with T=4000
but sample with as few as 25-100 steps, and the cosine schedule adapts
gracefully because its shape is preserved.
"""


# ════════════════════════════════════════════════════════════
# QUESTION 4 WRITTEN ANSWERS
# ════════════════════════════════════════════════════════════

"""
QUESTION 4a (4 pts):

The network predicts the noise ε rather than the clean image x₀ because:
(1) The noise ε is always drawn from a standard normal N(0,I) regardless
    of the input image or timestep, which gives the network a consistent,
    well-behaved target distribution to predict.
(2) Given the predicted noise, we can algebraically recover x₀ using the
    known relationship x_t = sqrt(ᾱ_t) * x₀ + sqrt(1-ᾱ_t) * ε.
(3) This parameterization naturally connects to denoising score matching —
    predicting ε is equivalent to estimating the score function ∇log p(x_t),
    which has strong theoretical grounding.
Ho et al. found empirically that this parameterization produces significantly
better sample quality than predicting x₀ or the mean directly.


QUESTION 4b (3 pts):

Consistency models learn a function f_θ(x_t, t) that maps any noisy
point x_t at any noise level t directly to the clean data x₀ in a
single forward pass. The key property they must satisfy is
self-consistency: for any two points x_t and x_t' that lie on the same
ODE trajectory (i.e., they came from the same clean image), the model
must output the same result: f_θ(x_t, t) = f_θ(x_t', t'). This
constraint allows the model to "shortcut" the entire iterative
denoising process.


QUESTION 4c (3 pts):

Advantage: Consistency models are dramatically faster at sampling —
generating images in 1-2 network evaluations instead of hundreds or
thousands. This makes them practical for real-time applications.

Disadvantage: One-step generation quality from consistency models is
slightly lower than full multi-step DDPM sampling (e.g., FID 3.55 vs
3.17 on CIFAR-10). Also, the best results (consistency distillation)
require a pre-trained diffusion model, so they don't fully eliminate
the training cost.
"""


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SOLUTION KEY — DDPM Forward Process Homework")
    print("=" * 60)

    print("\n Question 1: SNR Comparison")
    plot_snr_comparison()

    print("\n  Question 2: Visual Comparison")
    plot_schedule_visual_comparison()

    print("\n Question 3: Effect of T")
    plot_T_comparison()

    print("\n  Question 4: See written answers in this file (docstrings)")
    print("\nDone!")
