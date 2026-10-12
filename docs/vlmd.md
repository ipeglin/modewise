# Variational Latent Mode Decomposition (VLMD)

VLMD models a multichannel signal as a few shared **latent components** mixed into the channels by a sparse **loading** matrix. Each latent component is a sum of narrow-band **modes**, and every mode has one centre frequency shared by all latents: `X ≈ Σₖ Θ⁽ᵏ⁾A` (Morante & Rehman, 2025). It extends variational mode decomposition (Dragomiretskiy & Zosso, 2014) and its multivariate form (Rehman & Aftab, 2019) with a latent mixing model.

## How it works

VLMD solves its model with the alternating direction method of multipliers (ADMM; Boyd et al., 2011). Each iteration updates, in turn:

- **Loadings:** each channel's loadings are the lasso fit of that channel on the latent components (Tibshirani, 1996), with no intercept. Each latent's row of loadings is then divided by its largest absolute value when that is above 1, so every loading stays in [−1, 1]; the latent component takes up the scale in the next update.
- **Latent components:** each latent is fitted, in the frequency domain, to the part of the channels the other latents leave unexplained, while staying close to the sum of its modes. `reg_rho` sets that balance.
- **Modes:** each mode is a Wiener filter of its latent's residual around its centre frequency ωₖ, `1 / (1 + 4(α/ρ)(f − ωₖ)²)`, as in MVMD; larger α gives narrower modes.
- **Centre frequencies:** each ωₖ moves to the power-weighted mean frequency of its mode, pooled over latents.
- **Dual variables:** a step of size `tau` towards making each latent equal to the sum of its modes.

The run stops when `Σₖ((ωₖⁿ⁺¹ − ωₖⁿ)/ωₖⁿ)²` falls below `tolerance`, or after `max_iterations`.

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `modes` | mode × channel × sample | channel modes `U⁽ᵏ⁾ = Θ⁽ᵏ⁾A` (Eq. 7 in the paper) |
| `latent_modes` | mode × latent × sample | the latent modes `Θ⁽ᵏ⁾` |
| `loadings` | latent × channel | the sparse matrix `A`, with every value in [−1, 1] |

The channel modes add up to the latent fit `ZA`, not exactly to the signal. The difference is what the sparse fit leaves out, noise included. The reported centre frequencies are the ones used in the last mode update, in Hz.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `num_modes` | required | K |
| `num_latents` | one per channel | L, at most the number of channels (the model assumes L ≤ C) |
| `alpha` | 100 | bandwidth penalty α: larger values give narrower modes |
| `reg_lambda` | 0.01 | sparsity λ of the loadings, on the reference code's scale (see below) |
| `reg_rho` | 1 | ADMM penalty ρ. It sets how closely each latent must equal the sum of its modes, and it scales the bandwidth filter |
| `tau` | 0.9 | dual step, in [0, 1] |
| `tolerance` | 1e-3 | stop when the relative change of the centre frequencies reaches this |
| `max_iterations` | 1000 | |

The defaults are those of the authors' reference code. The paper's real-data runs used other values:
- exchange rates: α = 1000, ρ = 0.6, λ = 0.04
- electricity load: α = 5000, ρ = 5, λ = 0.5

**The scale of λ.** The paper writes the loading step as `‖X − ZA‖² + λ‖A‖₁`. The reference code calls scikit-learn's `Lasso` (Pedregosa et al., 2011) with `alpha = λ·10⁻³`, which minimises `(1/2T)‖x − Zᵀa‖² + alpha·‖a‖₁` for each channel. modewise uses the code's scale, so the published parameter values carry over. The paper's λ equals 2T·λ·10⁻³.

**Choosing λ for neuroimaging data.** The paper applies VLMD to synthetic, economic and power-grid data. On BOLD data the default λ gives dense loadings; raise it for sparse ones.

## Speed

The loading step is a lasso per channel, solved by coordinate descent on the Gram matrix (Friedman, Hastie & Tibshirani, 2010), in parallel over channels. With a small λ and correlated latents, as in BOLD, plain coordinate descent can need thousands of sweeps, so modewise also solves the optimality conditions exactly for the current signs, and accepts that solution only if the conditions hold.

With many latents, the latent components become band-limited after the first iterations and nearly collinear, and coordinate descent slows down. Each loading step therefore stops after 1000 sweeps, or earlier once the duality gap closes; a warning reports how many channels reached the limit.

Measured on one ds005237 run (434 channels, 493 samples, K = 6, 10 threads):

| Latents | Time |
|---|---|
| 40 | 1.1 s for the full run |
| 100 | 5.1 s for the full run |
| 200 | about 1 s per iteration |
| 434 (the default: one per channel) | about 25 s per iteration |

Choose `num_latents` well below the number of channels, as the paper does: 5 latents for 8 channels, and 30 for 251.

## Comparing with the reference code

modewise follows the paper's equations. It does not use the reference code ([Dmocrito/vlmd](https://github.com/Dmocrito/vlmd), GPL-3), which differs in a few places:

- It fits the loadings with `Lasso(tol=0.2)` and an intercept; modewise solves the lasso to convergence without an intercept, as in the paper's Eq. 10.
- Algorithm 1 in the paper bounds the loadings per channel column; the code and modewise bound them per latent row, a scale the latent component can absorb without changing `ZA`.
- The paper stops on the absolute change `Σ|Δω|²`; the code and modewise use the relative change, which does not depend on the frequency unit.
- The code divides frequencies by the sampling rate to get Hz; modewise multiplies by it.

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1–122. https://doi.org/10.1561/2200000016
- Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition. *IEEE Transactions on Signal Processing*, 62(3), 531–544. https://doi.org/10.1109/TSP.2013.2288675
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22. https://doi.org/10.18637/jss.v033.i01
- Morante, M., & Rehman, N. (2025). Latent mode decomposition. arXiv:2505.17797. https://arxiv.org/abs/2505.17797
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Rehman, N., & Aftab, H. (2019). Multivariate variational mode decomposition. *IEEE Transactions on Signal Processing*, 67(23), 6039–6052. https://doi.org/10.1109/TSP.2019.2951223
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B*, 58(1), 267–288.
