# Variational Latent Mode Decomposition (VLMD)

VLMD models a multichannel signal as a few shared **latent components** mixed into the channels by a sparse **loading** matrix. Each latent component is a sum of narrow-band **modes**, and every mode has one centre frequency shared by all latents: `X ≈ Σₖ Θ⁽ᵏ⁾A`.

Source: Morante and Rehman (2025), *Latent Mode Decomposition*, [arXiv:2505.17797](https://arxiv.org/abs/2505.17797).

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `modes` | mode × channel × sample | channel modes `U⁽ᵏ⁾ = Θ⁽ᵏ⁾A` (Eq. 7), what the paper evaluates |
| `latent_modes` | mode × latent × sample | the latent modes `Θ⁽ᵏ⁾` |
| `loadings` | latent × channel | the sparse matrix `A`, with every value in [−1, 1] |

The channel modes add up to the latent fit `ZA`, not exactly to the signal. The difference is what the sparse fit leaves out, noise included.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `num_modes` | required | K |
| `num_latents` | one per channel | L, at most the number of channels (the model assumes L ≤ C) |
| `alpha` | 100 | bandwidth penalty α: larger values give narrower modes |
| `reg_lambda` | 0.01 | sparsity λ of the loadings, on the reference code's scale (see below) |
| `reg_rho` | 1 | ADMM penalty ρ. It sets how closely each latent must equal the sum of its modes, and it scales the bandwidth filter to `1 + 4(α/ρ)(f − ωₖ)²` |
| `tau` | 0.9 | dual step, in [0, 1] |
| `tolerance` | 1e-3 | stop when `Σₖ((ωₖⁿ⁺¹ − ωₖⁿ)/ωₖⁿ)²` reaches this |
| `max_iterations` | 1000 | |

The defaults are the reference code's. The paper's real-data runs used other values:
- exchange rates: α = 1000, ρ = 0.6, λ = 0.04
- electricity load: α = 5000, ρ = 5, λ = 0.5

Its synthetic-data runs were tuned against the ground truth.

**The scale of λ.** The paper writes the loading step as `‖X − ZA‖² + λ‖A‖₁`. The reference code instead calls scikit-learn's `Lasso` with `alpha = λ·10⁻³`, which minimises `(1/2T)‖x − Zᵀa‖² + alpha·‖a‖₁` for each channel. modewise uses the code's scale, because the published parameter values were run with the code. The paper's λ equals 2T·λ·10⁻³.

## Where the paper and the reference code disagree

The reference code is [Dmocrito/vlmd](https://github.com/Dmocrito/vlmd) at commit `d5cbd2b`. modewise follows the paper's equations and our MVMD conventions. It does not port the reference code (GPL-3). Each disagreement below is decided by agreement with a second source.

| Topic | Paper | Reference code | modewise | Why |
|---|---|---|---|---|
| Loading step (Eq. 10) | exact minimiser, no intercept | `Lasso(tol=0.2)`, intercept fitted | exact minimiser, no intercept | The paper and convex theory agree: a minimiser does not depend on the solver. The code's result does: its output changed between scikit-learn 1.5 and 1.9 on identical input, so it is not a stable target. On synthetic data, recovery was similar either way (sometimes better, sometimes worse). |
| Keeping `|a| ≤ 1` | Algorithm 1 divides each **channel column** by `max{1, max(a_c)}` | divides each **latent row** by its largest absolute value, when above 1 | rows, absolute value | The paper justifies the bound by scale equivalence. Rescaling a latent row, with its latent signal rescaled inversely, leaves `ZA` unchanged; rescaling a channel column does not. |
| Stopping rule | `Σ|Δω|²` | `Σ(Δω/ω)²` | the code's | It is unit-free. In cycles per sample, the absolute rule stops too early at fMRI frequencies. Near-zero frequencies can make the relative rule slow to settle, but `max_iterations` bounds the run. |
| Dual step | Eq. 22 has none; Algorithm 1 uses τ = 0.9 | τ = 0.9 | τ = 0.9 | Algorithm 1 and the code agree |
| Frequencies in Hz | — | divides by the sampling rate | multiplies by it | The code's division is a bug that its default sampling rate of 1 hides |
| Reported centre frequencies | — | the ones used in the last filter step | the same | the same convention as our MVMD |

The paper's synthetic-data generator is not in the repository, so the synthetic tests use our own signals built from the same model.

## Evidence

| Check | Result |
|---|---|
| Loading step vs scikit-learn `Lasso(fit_intercept=False)`, 4 penalties, correlated latents | within 1e-9 |
| Full VLMD vs the reference with its loading step solved exactly, 3 cases (defaults; sparse with ρ, τ and λ changed; odd length with an iteration cap) | iteration counts identical; frequencies, loadings, latent and channel modes within 1e-9 (agreement measured below 1e-11) |
| Two tones mixed into four channels | frequencies within 0.002 cycles/sample; each channel mode peaks at its tone |
| Property tests on random signals | finite output, `|a| ≤ 1`, sorted frequencies within [0, Nyquist] |

**Speed.** The loading step is a lasso per channel, solved on the Gram matrix in parallel. With a small λ and correlated latents (as in BOLD), plain coordinate descent needed thousands of sweeps per channel per iteration. modewise therefore also solves the optimality conditions exactly for the current signs, and accepts the result only if those conditions hold.

When there are many latents, the latent components become band-limited after the first iterations. They are then nearly collinear, the loading step has no well-determined minimiser, and coordinate descent crawls. Each loading step therefore stops at 1000 sweeps, the reference's own budget (scikit-learn `max_iter=1000`), or earlier once the duality gap closes. A warning reports how many channels hit the limit.

Measured on one ds005237 run (434 channels, 493 samples, K = 6, 10 threads):

| Latents | Time |
|---|---|
| 40 | 1.1 s for the full run |
| 100 | 5.1 s for the full run |
| 200 | about 1 s per iteration |
| 434 (the default: one per channel) | about 25 s per iteration |

Choose `num_latents` well below the number of channels. The paper did the same: 5 latents for 8 channels, and 30 for 251.

## Open questions

- The paper tests VLMD on synthetic, economic and power-grid data only. Whether it suits neuroimaging data has not been tested; for the lab this is a research question.
- With λ = 0.01, the default, BOLD loadings come out dense. Sparse loadings need a larger λ, and no published guidance for neuro data exists yet.
