# %% [markdown]
# # Tutorial: DDPM Forward Process & Noise Schedules
# 
# **Course:** STAT 9100 — Denoising Diffusion Probabilistic Models  
# **Author:** Mengyan Jing  
# **Prerequisites:** `pip install numpy matplotlib`
# 
# ---
# 
# ## What You'll Learn
# 
# 1. How the DDPM **forward process** destroys an image with noise
# 2. The **closed-form shortcut** for jumping to any noise level
# 3. How the **noise schedule** (linear vs cosine) affects the process
# 4. How to read the **Signal-to-Noise Ratio (SNR)** curves
# 5. What the **reverse process** (generation) looks like conceptually
#
# After this tutorial, you'll be ready to do the homework,
# which asks you to implement a new schedule and explore further.
#
# ---

# %% [markdown]
# ## Part 1: Background — The Key Equation
#
# The DDPM forward process adds Gaussian noise to data over $T$ steps.
# The key insight is a **closed-form shortcut** — we can jump directly 
# from clean data $x_0$ to any noise level $t$ without stepping through 
# all intermediate steps:
#
# $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$
#
# Where:
# - $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ — cumulative product of the schedule
# - $\sqrt{\bar{\alpha}_t}$ — controls how much **original signal** remains
# - $\sqrt{1 - \bar{\alpha}_t}$ — controls how much **noise** is present
# - $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ — the Signal-to-Noise Ratio
#
# Let's implement this step by step.

# %%
# ── Imports ──
import numpy as np
import matplotlib.pyplot as plt

print("Setup complete!")

# %% [markdown]
# ## Part 2: Create a Test Image
#
# We'll use a colorful checkerboard pattern as our test image.
# In a real DDPM, this would be a photo (face, cat, etc.), but 
# a checkerboard lets us clearly see how noise destroys structure.

# %%
def make_test_image(size=128):
    """Creates a colorful 8x8 checkerboard image.
    
    Returns: numpy array, shape (size, size, 3), values in [0, 1]
    """
    block = size // 8
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.9, 0.3, 0.2]  # red-orange
            else:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = [0.2, 0.5, 0.9]  # blue
    return img

# Display it
x0 = make_test_image()
plt.figure(figsize=(3, 3))
plt.imshow(x0)
plt.title("Our test image x₀", fontsize=13)
plt.axis("off")
plt.show()
print(f"Image shape: {x0.shape}, pixel range: [{x0.min():.1f}, {x0.max():.1f}]")

# %% [markdown]
# ## Part 3: Define the Noise Schedules
#
# The noise schedule $\{\beta_t\}_{t=1}^{T}$ controls how quickly 
# we add noise. From the schedule, we derive $\bar{\alpha}_t$, 
# which tells us the total signal remaining at step $t$.
#
# We'll implement two schedules from the papers:
# - **Linear** (Ho et al. 2020): $\beta_t$ increases linearly from $10^{-4}$ to $0.02$
# - **Cosine** (Nichol & Dhariwal 2021): designed so $\bar{\alpha}_t$ follows a cosine curve

# %%
def linear_schedule(T=1000):
    """Linear beta schedule from Ho et al. (2020).
    
    betas go from 1e-4 to 2e-2 linearly.
    Returns alpha_bar: 1D array of length T.
    """
    betas = np.linspace(1e-4, 2e-2, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar


def cosine_schedule(T=1000):
    """Cosine schedule from Nichol & Dhariwal (2021).
    
    Designed so alpha_bar follows a cosine curve,
    providing smoother noise transitions.
    Returns alpha_bar: 1D array of length T.
    """
    s = 0.008  # small offset to avoid beta being too small near t=0
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]       # normalize so alpha_bar[0] ≈ 1
    ab = ab[1:]          # length T, for t = 1..T
    # derive betas and clip to avoid numerical issues
    betas = 1 - ab / np.concatenate([[1.0], ab[:-1]])
    betas = np.clip(betas, 0, 0.999)
    alpha_bar = np.cumprod(1 - betas)
    return alpha_bar


# Quick check: both should be length 1000, starting near 1 and ending near 0
ab_lin = linear_schedule(1000)
ab_cos = cosine_schedule(1000)
print(f"Linear:  alpha_bar[0]={ab_lin[0]:.6f}, alpha_bar[999]={ab_lin[999]:.8f}")
print(f"Cosine:  alpha_bar[0]={ab_cos[0]:.6f}, alpha_bar[999]={ab_cos[999]:.8f}")

# %% [markdown]
# ## Part 4: The Forward Process Function
#
# This is the core function — it implements the closed-form equation:
# $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon$$
#
# Notice how simple it is: just a weighted sum of the clean image and random noise.

# %%
def q_sample(x0, t, alpha_bar):
    """Forward process: sample x_t given x_0.
    
    This is the ONLY function needed for the forward process.
    No iterative stepping — we jump directly to timestep t.
    
    Args:
        x0:        clean image, shape (H, W, 3), values in [0, 1]
        t:         integer timestep (0-indexed)
        alpha_bar: 1D array of cumulative products, length T
    
    Returns:
        noisy image x_t, clipped to [0, 1] for display
    """
    eps = np.random.randn(*x0.shape)                          # ε ~ N(0, I)
    signal_part = np.sqrt(alpha_bar[t]) * x0                  # √ᾱ_t · x₀
    noise_part  = np.sqrt(1 - alpha_bar[t]) * eps             # √(1-ᾱ_t) · ε
    xt = signal_part + noise_part                              # x_t
    return xt.clip(0, 1)  # clip for valid pixel display

# %% [markdown]
# ## Part 5: Demo 1 — Watch Diffusion Destroy an Image
#
# Let's apply the forward process at several timesteps to see 
# how the image progressively gets noisier.

# %%
np.random.seed(42)
x0 = make_test_image()
T = 1000
alpha_bar = linear_schedule(T)

timesteps = [0, 50, 150, 300, 500, 750, 999]

fig, axes = plt.subplots(1, len(timesteps), figsize=(16, 3))
for ax, t in zip(axes, timesteps):
    xt = q_sample(x0, t, alpha_bar)
    ax.imshow(xt)
    ax.set_title(f"t = {t}", fontsize=12, fontweight='bold')
    ax.axis("off")

fig.suptitle("Forward Process: x₀ → x_T (linear schedule)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What to notice:**
# - At $t=0$: clean image, no noise
# - At $t=50$: barely noticeable change (SNR is still very high)
# - At $t=300$: colors fading, edges blurring
# - At $t=500$: mostly noise, structure barely visible
# - At $t=999$: pure noise — completely destroyed
#
# The model's job is to learn to **reverse** this process.

# %% [markdown]
# ## Part 6: Demo 2 — Reverse Process (Generation)
#
# Now let's show what generation looks like: structure **emerging**
# from pure noise. In a real DDPM, a trained neural network would
# denoise step by step. Here we simulate the visual effect by showing
# the same image at decreasing noise levels.

# %%
np.random.seed(42)
x0 = make_test_image()
alpha_bar = linear_schedule(1000)

# Show timesteps in REVERSE order: noise → clean
timesteps_reverse = [999, 750, 500, 300, 150, 50, 0]

fig, axes = plt.subplots(1, len(timesteps_reverse), figsize=(16, 3))
for idx, (ax, t) in enumerate(zip(axes, timesteps_reverse)):
    if t == 0:
        xt = x0.copy()
    else:
        np.random.seed(42)  # same noise for consistency
        eps = np.random.randn(*x0.shape)
        xt = (np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps).clip(0, 1)
    ax.imshow(xt)
    ax.set_title(f"step {idx}\n(t={t})", fontsize=11, fontweight='bold')
    ax.axis("off")

fig.suptitle("Reverse Process (generation): noise → image emerges step by step",
             fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What to notice:**
# - Step 0: pure random noise — no structure at all
# - Steps 1–3: vague blobs of color start appearing
# - Steps 4–5: edges and checkerboard pattern become clear
# - Step 6: fully clean, sharp image
#
# In a real DDPM, the neural network has **never seen** the target image.
# It generates a **brand new** image by iteratively denoising pure noise.
# The structure emerges because the network has learned the statistics 
# of the training data.

# %% [markdown]
# ## Part 7: Demo 3 — Why the Noise Schedule Matters
#
# Not all schedules are created equal. Let's compare the **linear**
# schedule (Ho et al. 2020) with the **cosine** schedule (Nichol & 
# Dhariwal 2021) by plotting $\bar{\alpha}_t$ and $\text{SNR}(t)$.

# %%
def compute_snr(alpha_bar):
    """Signal-to-Noise Ratio: SNR(t) = ᾱ_t / (1 - ᾱ_t)"""
    return alpha_bar / (1 - alpha_bar + 1e-12)

T = 1000
ab_lin = linear_schedule(T)
ab_cos = cosine_schedule(T)

snr_lin = compute_snr(ab_lin)
snr_cos = compute_snr(ab_cos)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: alpha_bar curves
axes[0].plot(ab_lin, label="Linear", linewidth=2, color="#ef4444")
axes[0].plot(ab_cos, label="Cosine", linewidth=2, color="#3b82f6")
axes[0].set_xlabel("Timestep t", fontsize=12)
axes[0].set_ylabel("$\\bar{\\alpha}_t$ (signal fraction)", fontsize=12)
axes[0].set_title("How much signal remains?", fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Right: SNR curves (log scale)
axes[1].plot(snr_lin, label="Linear", linewidth=2, color="#ef4444")
axes[1].plot(snr_cos, label="Cosine", linewidth=2, color="#3b82f6")
axes[1].set_yscale("log")
axes[1].set_xlabel("Timestep t", fontsize=12)
axes[1].set_ylabel("SNR(t)  [log scale]", fontsize=12)
axes[1].set_title("Signal-to-Noise Ratio", fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# **Key observations:**
# - **Linear** (red): $\bar{\alpha}_t$ drops to ~0 by step 750. 
#   The last 25% of timesteps are wasted — just shuffling pure noise.
# - **Cosine** (blue): $\bar{\alpha}_t$ decreases gradually across 
#   all 1000 steps. Every timestep does useful work.
# - The **SNR plot** (right) shows this on a log scale: cosine 
#   provides a much smoother transition from high SNR to low SNR.
#
# This is why Nichol & Dhariwal's cosine schedule produces better 
# images, especially at lower resolutions (32×32, 64×64).

# %% [markdown]
# ## Part 8: Visual Comparison — Linear vs Cosine Side by Side
#
# Let's see the difference visually: same image, same noise,
# different schedules.

# %%
np.random.seed(123)
x0 = make_test_image()
T = 1000
ab_lin = linear_schedule(T)
ab_cos = cosine_schedule(T)

def q_sample_with_schedule(x0, t, ab):
    """Same as q_sample but takes any alpha_bar array."""
    eps = np.random.randn(*x0.shape)
    return (np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps).clip(0, 1)

compare_t = [200, 400, 600, 800]
fig, axes = plt.subplots(2, len(compare_t), figsize=(14, 5))

for col, t in enumerate(compare_t):
    np.random.seed(col)  # same noise for fair comparison
    img_lin = q_sample_with_schedule(x0, t, ab_lin)
    np.random.seed(col)  # reset seed — exact same noise
    img_cos = q_sample_with_schedule(x0, t, ab_cos)

    axes[0, col].imshow(img_lin)
    axes[0, col].set_title(f"t={t}", fontsize=11, fontweight='bold')
    axes[0, col].axis("off")

    axes[1, col].imshow(img_cos)
    axes[1, col].set_title(f"t={t}", fontsize=11)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Linear", fontsize=12, fontweight='bold', rotation=0, labelpad=50)
axes[1, 0].set_ylabel("Cosine", fontsize=12, fontweight='bold', rotation=0, labelpad=50)
fig.suptitle("Same noise, different schedules — cosine preserves structure longer",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What to notice:**
# - At $t=400$: Linear is almost pure noise; Cosine still shows 
#   clear checkerboard structure
# - At $t=600$: Linear is completely destroyed; Cosine still has 
#   some visible pattern
# - This means the cosine schedule gives the neural network more 
#   "useful" denoising work at each timestep

# %% [markdown]
# ## Part 9: Key Functions Summary
#
# Here's a summary of all the functions we built. These same 
# functions appear in the **homework starter code** (`hw_starter.py`),
# so you're already familiar with them!
#
# | Function | What it does |
# |----------|-------------|
# | `make_test_image()` | Creates a colored checkerboard |
# | `linear_schedule(T)` | Returns $\bar{\alpha}_t$ for linear schedule |
# | `cosine_schedule(T)` | Returns $\bar{\alpha}_t$ for cosine schedule |
# | `q_sample(x0, t, alpha_bar)` | Forward process: adds noise to get $x_t$ |
# | `compute_snr(alpha_bar)` | Computes $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$ |
#
# ## Connection to the Homework
#
# In the homework, you will:
# 1. **Implement a new schedule** (sqrt) using the same pattern as above
# 2. **Visualize** and compare your schedule against linear and cosine
# 3. **Explore** how the number of total steps $T$ affects the schedule
# 4. **Answer conceptual questions** about the training loss and consistency models
#
# The homework starter code (`hw_starter.py`) reuses the exact same 
# helper functions from this tutorial, so everything should feel familiar.

# %% [markdown]
# ## Part 10: Quick Recap — The Full DDPM Pipeline
#
# What we covered in this tutorial is **Step 1** (Forward Process) of the 
# DDPM pipeline. Here's how it fits into the bigger picture:
#
# ```
# TRAINING:
#   1. Take clean image x₀ from dataset
#   2. Pick random timestep t ~ Uniform(1, T)
#   3. Add noise: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε     ← THIS TUTORIAL
#   4. Network predicts noise: ε̂ = ε_θ(x_t, t)
#   5. Loss = ||ε - ε̂||²  (MSE)
#
# GENERATION (sampling):
#   1. Start from pure noise x_T ~ N(0, I)
#   2. For t = T down to 1:
#        Use ε_θ(x_t, t) to compute x_{t-1}
#   3. Return x₀ (generated image!)
# ```
#
# The forward process (Step 3) is what we implemented here. 
# It requires **no training** — it's pure math. The neural network 
# only appears in Steps 4-5 (training) and Step 2 of generation.
#
# ---
# 
# **End of tutorial.** Now try the homework! 🎉
