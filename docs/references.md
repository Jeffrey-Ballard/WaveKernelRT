# Algorithm references

This page records the primary public references for specialized algorithms in
WaveKernelRT. General numerical operations such as FFTs, windowing, and
conventional beamforming are documented through their mathematical conventions
and implementation choices rather than attributed to a single source.

## Noise-normalized energy detection

### Abraham (2019)

Douglas A. Abraham, *Underwater Acoustic Signal Processing: Modeling,
Detection, and Estimation*, 1st ed., Springer, Cham, 2019, Chapter 9,
“Detecting Signals with Unknown Form: Energy Detectors,” pp. 621–741.
[Publisher book record](https://link.springer.com/book/10.1007/978-3-319-92983-5) ·
[Publisher chapter record](https://link.springer.com/chapter/10.1007/978-3-319-92983-5_9)

Chapter 9 provides the detection and background-normalization basis used by
WaveKernelRT:

- Section 9.1, p. 622, Eq. (9.2), relates energy to squared Fourier-transform
  magnitudes and supports forming an energy statistic in the frequency domain.
- Section 9.2.6.1, pp. 646–647, Eqs. (9.84) and (9.88), analyzes
  noise-normalized energy for a Gaussian random signal with unknown spectral
  shape in a known band. The local and asymptotic optimality results require
  the known-noise and independent-bin assumptions of that derivation.
- Section 9.3, p. 683, Eqs. (9.190)–(9.191), divides each frequency-bin power
  by its estimated noise power and then sums the linear normalized powers.
  This is the direct basis for WaveKernelRT's normalization-before-integration
  order. WaveKernelRT additionally applies frequency-bin-width weights to
  integrate PSD ratios over hertz.
- Section 9.3.4.2, p. 697, illustrates averaging the linear normalized powers
  before conversion to decibels. WaveKernelRT likewise converts only after
  linear aggregation, but its bin-width-weighted integral is not numerically
  the same statistic as Abraham's unweighted mean.
- Section 9.3.1, pp. 683–685, motivates nearby beams as auxiliary observations
  for broadband backgrounds and sharing observations across nearby frequencies
  when the noise spectrum varies slowly. Equations (9.192)–(9.194) support an
  arithmetic mean of linear auxiliary powers under their independent,
  identically distributed exponential model.
- Section 9.3.4.1, p. 693, Eqs. (9.220)–(9.222), extends fixed weighted-power
  background estimates to correlated auxiliary observations. This extension
  does not remove the separate independence assumptions between a test cell
  and its background estimate or among terms in the final detector statistic.

The match to WaveKernelRT is deliberately limited. The implementation's
centered finite time window is not the exponential averager of Abraham
Eq. (9.205). Its decibel-domain frequency convolution, bearing-neighborhood
median, guarded clip-and-replace filters, integration weights, and ranked
output are implementation choices rather than algorithms specified by the
book. In particular, the clip-and-replace rules need the separate Hampel-family
comparison below.

The variance-equivalent length for Abraham's exponential averager in
Eq. (9.210) is not an exact false-alarm threshold model; the limitations are
illustrated in Section 9.3.4.2, pp. 697–698, Fig. 9.16. WaveKernelRT therefore
does not infer a calibrated false-alarm probability or detector threshold from
its background window length or from the cited idealized optimality results.

## Background estimation

### Pearson et al. (2016)

R. K. Pearson, Y. Neuvo, J. Astola, and M. Gabbouj, “Generalized Hampel
Filters,” *EURASIP Journal on Advances in Signal Processing*, article 87,
2016. [doi:10.1186/s13634-016-0383-6](https://doi.org/10.1186/s13634-016-0383-6)

Equations (2)–(4) define the moving median, median absolute deviation (MAD)
scale estimate, and standard Hampel decision rule. The paper also interprets
the median filter as the zero-threshold limit and discusses cascades of median
and Hampel-family filters. WaveKernelRT's frequency-axis clip-and-replace
stages share the decision-filter pattern—retain an ordinary value and replace
an identified peak with a local reference—but are not implementations of the
paper's standard, recursive, or weighted Hampel filters. WaveKernelRT uses a
guarded moving mean, a fixed one-sided decibel threshold, and that mean as the
replacement; a standard Hampel filter uses a moving median, a MAD-scaled
two-sided threshold, and the median as the replacement.

The exact estimator sequence and the relationship between its median and
clip-and-replace operations are described in [Band-energy
detection](concepts/band-energy-detection.md#relationship-to-hampel-filtering).

## Dominant Mode Rejection

### Cox et al. (2005)

H. Cox, A. Pezeshki, L. L. Scharf, O. Besson, and H. Lai, “Multi-Rank
Adaptive Beamforming with Linear and Quadratic Constraints,” *Conference
Record of the Thirty-Ninth Asilomar Conference on Signals, Systems and
Computers*, pp. 1692–1697, 2005.
[Public copy](https://www.researchgate.net/publication/239920591_Multi-Rank_Adaptive_Beamforming_with_Linear_and_Quadratic_Constraints)

Section II-E and equations (20)–(23) are the closest published basis for the
WaveKernelRT weight construction: a truncated covariance, diagonal loading,
a norm constraint, unity-response normalization, and loading that varies with
the steering direction. Cox et al. separate a noise-estimate load from the
additional norm-control load; WaveKernelRT searches for one total load.

### Wage and Buck (2010)

K. E. Wage and J. R. Buck, “Performance Analysis of Dominant Mode Rejection
Beamforming,” *Proceedings of the 20th International Congress on Acoustics*,
Sydney, Australia, 2010.
[Public PDF](https://www.acoustics.asn.au/conference_proceedings/ICA2010/cdrom-ICA2010/papers/p412.pdf)

Equation (5) supplies the ordinary equal-weight sample covariance used when
WaveKernelRT completes covariance startup. The paper also documents
conventional DMR, explicit rank estimators, and the risks of signal
cancellation and steering mismatch.

### Hulbert and Wage (2022)

C. C. Hulbert and K. E. Wage, “Random Matrix Theory Predictions of Dominant
Mode Rejection Beamformer Performance,” *IEEE Open Journal of Signal
Processing*, vol. 3, pp. 229–245, 2022.
[doi:10.1109/OJSP.2022.3185937](https://doi.org/10.1109/OJSP.2022.3185937)

Equations (5)–(7) give the sample covariance, its eigenbasis representation,
and the white-noise-gain definition used to derive WaveKernelRT's reduced
eigenbasis WNG calculation. Section V-C, equations (72)–(73), gives the
strong-interferer, matched-rank, signal-free-training approximation
$\mathcal{L}\approx1-R/T$ used to derive WaveKernelRT's nominal covariance
training reference.

### Campos Anchieta and Buck (2022)

D. Campos Anchieta and J. R. Buck, “Improving the Robustness of the Dominant
Mode Rejection Beamformer With Median Filtering,” *IEEE Access*, vol. 10,
pp. 120146–120154, 2022.
[doi:10.1109/ACCESS.2022.3221954](https://doi.org/10.1109/ACCESS.2022.3221954)

Section III explains why practical DMR implementations often conservatively
overestimate dominant rank and analyzes the accompanying bias and robustness
costs. It supports the motivation for a conservative rank choice, but not
WaveKernelRT's particular energy-threshold-plus-one rule.

### Hulbert and Wage (2025)

C. C. Hulbert and K. E. Wage, “Random Matrix Theory Predictions of Dominant
Mode Rejection SINR Loss Due to Signal in the Training Data,” accepted author
manuscript, *IEEE Open Journal of Signal Processing*, 2025.
[doi:10.1109/OJSP.2025.3578812](https://doi.org/10.1109/OJSP.2025.3578812)

This paper treats signal-containing training data. Section V analyzes
overestimated rank, including the additional noise eigenvectors admitted into
the dominant subspace. It limits interpretation of the nominal SINR-loss input:
training requirements can depend on desired-signal strength and subspace
geometry when the desired signal is present. It does not prescribe
WaveKernelRT's rank heuristic or training policy.

### Implementation mapping

The WaveKernelRT element-space implementation is configured by
`DMRElementConfig` and executed by `FFTtoDMRElement_Stage`. The table separates
published foundations from local policy so that a citation does not imply an
exact reproduction of a paper's algorithm.

| Implementation choice | Literature relationship |
| --- | --- |
| Covariance contributions $xx^H$ and the completed startup mean | Wage and Buck (2010), Eq. (5); Hulbert and Wage (2022), Eq. (5) |
| Reference budget $T_{\mathrm{ref}}=\lceil D_{\max}/(1-10^{-\delta/10})\rceil$ | Algebraic rearrangement of Hulbert and Wage (2022), Sec. V-C, Eqs. (72)–(73), with the engineering substitution of `Dmax` for interference rank $R$; `sinr_loss_dB` supplies $\delta$ and defaults to 1 dB |
| EMA coefficient $\alpha=2/(T_{\mathrm{ref}}+1)$ | WaveKernelRT policy that makes $(2-\alpha)/\alpha=T_{\mathrm{ref}}$ under an independent stationary variance-equivalence calculation; not a DMR SINR result |
| Retaining eigenvectors with the largest eigenvalues | Cox et al. (2005), Eq. (23); Hulbert and Wage (2022), Sec. III-A |
| $U_D\Lambda_DU_D^H+\epsilon I$, unity-response normalization, and beam-dependent WNG loading | Cox et al. (2005), Sec. II-E and Eqs. (20)–(23), specialized to one steering vector |
| Eigenbasis WNG evaluation | Algebraic reduction of Cox et al. (2005), Eqs. (22)–(23), using the WNG definition in Hulbert and Wage (2022), Eq. (7) |
| Rank from cumulative eigenvalue energy | WaveKernelRT heuristic; the cited papers do not specify this estimator |
| One extra retained mode | Exact rule is local; Campos Anchieta and Buck (2022), Sec. III, and Hulbert and Wage (2025), Sec. V, support analyzing conservative rank overestimation and its costs |
| Reusing one eigendecomposition for beams with the same active aperture | Algebraic optimization that preserves the per-beam calculation |

The implemented equations, startup behavior, and limitations are described in
[Beamforming](concepts/beamforming.md#dominant-mode-rejection).
