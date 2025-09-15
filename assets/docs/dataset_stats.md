# Dataset Statistics

## Purpose

Compare three datasets used in a cross-attention MTL pipeline:

* `angio_det`: angiography detection images
* `angio_seg`: angiography segmentation images
* `retina_seg`: retinography vessel-segmentation images

The audit quantifies and visualizes distributional differences to assess domain shift risk and guide harmonization.

---

## Figures and how to read them

### 1) Embedding centroid distances heatmap

File: `figs/embedding_centroid_distances.png`
What it shows: pairwise Euclidean distances between dataset means in a deep-feature space extracted by a fixed encoder. Larger values indicate stronger distributional separation in representation space. Deep features are a standard proxy for dataset similarity; distances relate to metrics like FID/KID that compare feature distributions rather than pixels \[4–5].
Interpretation here: `retina_seg` is far from both angiography sets. `angio_det` and `angio_seg` are closer, but not identical.

### 2) Metric panels: ECDF + boxplots

File: `figs/panels_metrics.png`
Per-feature left panel = ECDF, right panel = boxplot.

* **Mean intensity**: ECDFs and boxplots show `retina_seg` has lower mean than both angiography sets.
* **Std. dev.**: `retina_seg` shows higher contrast dispersion.
* **Laplacian variance (lap\_var)**: markedly higher for `retina_seg`, indicating more high-frequency content or sharper edges; Laplacian-based focus measures are widely used for sharpness/blur assessment \[8–9].
* **Edge density (Canny)**: `retina_seg` sits between the two angiography sets; Canny edge detector is a classical, well-characterized gradient method \[10].
* **Power-spectrum slope (ps\_slope)**: all follow natural-image-like \~1/f^α behavior; shifts in slope reflect differences in spatial frequency content \[6–7].
  Why ECDFs/boxplots: ECDFs are nonparametric views of full distributions \[11]; boxplots summarize location/scale.

### 3) QQ-plot panels

File: `figs/qq_panels.png`
What it shows: quantile–quantile comparisons for selected features. Points on the 45° line imply matched distributions; systematic deviations expose location/scale or tail differences. QQ-plots are standard to compare distributions beyond means/variances \[12].
Interpretation here:

* `retina_seg` vs `angio_seg` for **mean**: strong curvature → different location and scale.
* `retina_seg` vs `angio_seg` for **lap\_var**: `retina_seg` dominates in upper quantiles → sharper/high-frequency content.
* `angio_det` vs `angio_seg`: closer but still systematic differences.

### 4) UMAP of embeddings

File: `figs/umap_embeddings.png`
What it shows: 2-D UMAP of deep embeddings with dataset colors. UMAP preserves local neighborhood structure and more global geometry than t-SNE for many datasets, at lower cost \[1].
Interpretation here: three clear clusters; `angio_det` and `angio_seg` partially overlap but are separable; `retina_seg` forms a distinct manifold.

---

## Results summary

* **Global representation gap**: Large `retina_seg` vs angiography distances in embedding space and well-separated UMAP clusters indicate a substantial domain shift. This aligns with known modality/domain effects in medical imaging and their impact on model transfer \[13–16].
* **Low-level statistics**: `retina_seg` shows lower mean, higher std., much higher Laplacian variance, and different edge densities. These point to contrast, sharpness, and texture differences beyond simple intensity scaling \[8–10].
* **Frequency structure**: Differences in **ps\_slope** imply distinct distributions of spatial scales; such spectral shifts are known correlates of image acquisition and content differences \[6–7].
* **Angio sets relationship**: `angio_det` and `angio_seg` are closer than either is to `retina_seg`, but QQ and ECDFs still reveal measurable shifts.

**Implication for your MTL plan**
Before or during cross-domain training, harmonize first- and second-order luminance statistics, mitigate sharpness/spectrum gaps, and consider feature-space alignment. Evidence across medical imaging shows domain shift degrades performance and that transfer benefits depend on representation alignment rather than raw pixel similarity \[13,15–16].

---

## Figure-by-figure checklist for future runs

* If **centroid distances** increase after data updates, investigate acquisition or preprocessing changes.
* If **ECDFs** misalign for `mean`/`std`, apply intensity standardization per dataset before joint training.
* If **lap\_var** and **ps\_slope** diverge, address blur/spectral differences via deblurring, unsharp masking with care, or matched filtering; alternatively, enforce spectrum-aware augmentations.
* If **UMAP** clusters separate further, expect more aggressive domain adaptation or multi-domain normalization to be required.

---

## References

\[1] McInnes, Healy, Melville. “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.” 2018. ([arXiv][1])

\[2] Rizzo, Székely. “Energy distance.” WIREs Comp Stats, 2016. ([Wiley Online Library][2])

\[3] Gretton et al. “A Kernel Two-Sample Test.” JMLR, 2012. ([Journal of Machine Learning Research][3])

\[4] Heusel et al. “GANs trained by a Two Time-Scale Update Rule… (FID).” NeurIPS, 2017. ([NeurIPS Papers][4])

\[5] Bińkowski et al. “Demystifying MMD GANs (Kernel Inception Distance).” 2018. ([arXiv][5])

\[6] Ruderman, Bialek. “Statistics of Natural Images: Scaling in the Woods.” 1994. ([swh.princeton.edu][6])

\[7] van der Schaaf, van Hateren. “Modelling the power spectra of natural images.” Vision Research, 1996. ([ScienceDirect][7])

\[8] Pertuz, Puig, Garcia. “Analysis of focus measure operators for shape-from-focus.” Pattern Recognition, 2013. ([ISP UTB][8])

\[9] Memon et al. “Image Quality Assessment for Performance Evaluation of Focus Measure Operators.” 2016. ([arXiv][9])

\[10] Canny. “A Computational Approach to Edge Detection.” IEEE PAMI, 1986. ([cse.ust.hk][10])

\[11] OpenTURNS docs: “Empirical cumulative distribution function.” ([openturns.github.io][11])

\[12] NIST/SEMATECH e-Handbook: “Quantile-Quantile Plot.” ([NIST ITL][12])

\[13] Zhang et al. “Generalizing Deep Learning for Medical Image Analysis…” 2020. ([PMC][13])

\[14] Guan, Liu. “Domain Adaptation for Medical Image Analysis: A Survey.” 2022. ([PMC][14])

\[15] Raghu et al. “Transfusion: Understanding Transfer Learning for Medical Imaging.” NeurIPS, 2019. ([NeurIPS Papers][15])

\[16] Kilim et al. “Physical imaging parameter variation drives domain shift.” Sci Reports, 2022. ([Nature][16])

---

## Reproduction

All figures are created by `dataset_similarity_audit.py`:

* per-image metrics → `metrics.csv`
* texture metrics → `glcm_features.csv`
* embeddings → `embeddings.csv`
* statistical tests → `stats_*.csv`
* figures → `figs/*.png`



[1]: https://arxiv.org/abs/1802.03426?utm_source=chatgpt.com "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
[2]: https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.1375?utm_source=chatgpt.com "Energy distance - Rizzo - 2016"
[3]: https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf?utm_source=chatgpt.com "A Kernel Two-Sample Test"
[4]: https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium?utm_source=chatgpt.com "GANs Trained by a Two Time-Scale Update Rule ..."
[5]: https://arxiv.org/abs/1801.01401?utm_source=chatgpt.com "[1801.01401] Demystifying MMD GANs"
[6]: https://swh.princeton.edu/~wbialek/rome/refs/ruderman%2Bbialek_94.pdf?utm_source=chatgpt.com "Statistics of Natural Images: Scaling in the Woods"
[7]: https://www.sciencedirect.com/science/article/pii/0042698996000028?utm_source=chatgpt.com "Modelling the Power Spectra of Natural Images: Statistics ..."
[8]: https://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf?utm_source=chatgpt.com "Analysis of focus measure operators in shape - GitHub Pages"
[9]: https://arxiv.org/abs/1604.00546?utm_source=chatgpt.com "Image Quality Assessment for Performance Evaluation of Focus Measure Operators"
[10]: https://www.cse.ust.hk/~quan/comp5421/notes/canny1986.pdf?utm_source=chatgpt.com "A Computational Approach to Edge Detection"
[11]: https://openturns.github.io/openturns/latest/theory/data_analysis/empirical_cdf.html?utm_source=chatgpt.com "Empirical cumulative distribution function - OpenTURNS"
[12]: https://www.itl.nist.gov/div898/handbook/eda/section3/qqplot.htm?utm_source=chatgpt.com "1.3.3.24. Quantile-Quantile Plot"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676/?utm_source=chatgpt.com "Generalizing Deep Learning for Medical Image ..."
[14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/?utm_source=chatgpt.com "Domain Adaptation for Medical Image Analysis: A Survey"
[15]: https://papers.neurips.cc/paper/8596-transfusion-understanding-transfer-learning-for-medical-imaging.pdf?utm_source=chatgpt.com "Understanding Transfer Learning for Medical Imaging"
[16]: https://www.nature.com/articles/s41598-022-23990-4?utm_source=chatgpt.com "Physical imaging parameter variation drives domain shift"
