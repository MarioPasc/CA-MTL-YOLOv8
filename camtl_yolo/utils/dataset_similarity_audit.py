#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset similarity audit for cross-domain MTL (retinography ↔ angiography).

Computes per-image statistics, classical texture features, frequency slopes,
deep embeddings, and runs dataset-wise two-sample tests (KS, Mann–Whitney,
Cliff's delta, Energy test, MMD). Also computes FID/KID between datasets
(optional, requires torch + torchvision).

References:
- Fréchet Inception Distance (FID): Heusel et al., NeurIPS 2017.  # arXiv:1706.08500
- Kernel Inception Distance (KID): Bińkowski et al., 2018.       # arXiv:1801.01401
- MMD two-sample test: Gretton et al., JMLR 2012.                 # jmlr.org v13/gretton12a
- Energy distance & tests: Rizzo & Székely, WIREs Comp Stat 2016; SJS 2013.
- Natural-image 1/f spectra: van der Schaaf & van Hateren, Vis Res 1996; Torralba notes.
- Haralick / GLCM + gray-level invariance: Haralick et al., 1973; Carrillo-Perez et al., 2019.

This script creates:
  - metrics.csv                  # per-image scalar metrics
  - glcm_features.csv            # per-image texture features
  - embeddings.csv               # per-image embeddings (if enabled)
  - stats_report.csv             # dataset-wise tests and distances
  - figs/*.png                   # ECDFs, QQ plots, power spectra, heatmaps, UMAP

Usage:
  python dataset_similarity_audit.py \
      --det_dir /path/angio_detection_images \
      --seg_dir /path/angio_seg_images \
      --retina_seg_dir /path/retina_seg_images \
      --out_dir ./audit_out \
      --compute_embeddings true \
      --compute_fid true
python camtl_yolo/utils/dataset_similarity_audit.py \
      --det_dir /media/mpascual/PortableSSD/Coronariografías/processed_data/angiography/detect/images \
      --seg_dir /media/mpascual/PortableSSD/Coronariografías/processed_data/angiography/segment/images \
      --retina_seg_dir /media/mpascual/PortableSSD/Coronariografías/processed_data/retinography/images \
      --out_dir /media/mpascual/PortableSSD/Coronariografías/stats_processed_data \
      --max_images_per_set 500 \
      --device cuda:0
"""
from __future__ import annotations
import logging, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd # type: ignore
from scipy import stats # type: ignore
from scipy.linalg import sqrtm # type: ignore
from scipy.fft import fft2, fftshift # type: ignore
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import sobel, laplace
from skimage.feature import canny
from skimage.util import img_as_ubyte
from skimage.measure import shannon_entropy
from skimage.feature.texture import graycomatrix, graycoprops
from tqdm import tqdm # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
try:
    import umap  # type: ignore
    UMAP_OK = True
except Exception:
    UMAP_OK = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: embeddings and FID
try:
    import torch
    import torchvision.transforms as T # type: ignore
    import torchvision.models as models # type: ignore
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# --------------------------- Config ------------------------------------------
@dataclass(frozen=True)
class Config:
    det_dir: Path
    seg_dir: Path
    retina_seg_dir: Path
    out_dir: Path
    max_images_per_set: Optional[int] = None
    compute_embeddings: bool = True
    compute_fid: bool = True
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    glcm_distances: Tuple[int, ...] = (1, 2, 4)      # pixels
    glcm_angles: Tuple[float, ...] = (0.0, np.pi/4, np.pi/2, 3*np.pi/4)
    glcm_levels: int = 32                            # quantization levels
    embed_image_size: int = 224
    random_state: int = 17

# --------------------------- Utilities ---------------------------------------

def configure_matplotlib() -> None:
    """
    Configure matplotlib and scienceplots with LaTeX and requested typography.
    Falls back gracefully if LaTeX or scienceplots are not available.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])  # base science style
    except Exception as e:
        logging.warning("scienceplots not available: %s. Continuing with default style.", e)

    # Requested typography
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })
    # LaTeX text rendering
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s. Falling back to non-LaTeX text.", e)
        plt.rcParams['text.usetex'] = False
def list_images(d: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p for p in d.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files

def imread_uint8_gray(p: Path) -> np.ndarray:
    """Read image, convert to grayscale uint8 [0,255]. Handles RGB/RGBA/gray."""
    img = imread(str(p))
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    # normalize to [0,1] if not already, then to uint8
    img = rescale_intensity(img, in_range='image', out_range=(0, 1))
    img_u8 = img_as_ubyte(img)
    return img_u8

def basic_stats(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(np.float32) / 255.0
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    sk = float(stats.skew(x.ravel(), bias=False))
    ku = float(stats.kurtosis(x.ravel(), fisher=True, bias=False))
    ent = float(shannon_entropy((x*255).astype(np.uint8), base=2))
    return {"mean": m, "std": s, "skew": sk, "kurtosis": ku, "entropy": ent}

def gradient_edge_stats(x_u8: np.ndarray) -> Dict[str, float]:
    x = x_u8.astype(np.float32) / 255.0
    sob = sobel(x)
    g_p50 = float(np.percentile(sob, 50))
    g_p90 = float(np.percentile(sob, 90))
    lap = laplace(x)
    sharp = float(np.var(lap))
    edges = canny(x, sigma=1.0)
    edge_density = float(np.mean(edges))
    return {"sobel_p50": g_p50, "sobel_p90": g_p90, "lap_var": sharp, "edge_density": edge_density}

def radial_power_spectrum_slope(x_u8: np.ndarray) -> Dict[str, float]:
    """Fit log power vs log freq; return slope and intercept."""
    x = x_u8.astype(np.float32) / 255.0
    # remove DC trend
    x = x - np.mean(x)
    F = fftshift(fft2(x))
    P = np.abs(F)**2
    h, w = x.shape
    cy, cx = h//2, w//2
    y, xg = np.ogrid[:h, :w]
    r = np.hypot(y - cy, xg - cx)
    r = r.astype(np.int32)
    maxr = np.min([cy, cx])
    radial_mean = np.bincount(r.ravel(), P.ravel())[:maxr] / np.maximum(1, np.bincount(r.ravel())[:maxr])
    freqs = np.arange(1, len(radial_mean), dtype=np.float32)
    ps = radial_mean[1:].astype(np.float32) + 1e-12
    X = np.log(freqs)
    Y = np.log(ps)
    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
    return {"ps_slope": float(slope), "ps_intercept": float(intercept)}

def glcm_features(x_u8: np.ndarray, cfg: Config) -> Dict[str, float]:
    """Gray-level invariant GLCM: quantize to cfg.glcm_levels over image min..max."""
    x = x_u8.astype(np.uint8)
    # Scale to [0, levels-1]
    xq = (x.astype(np.float32) / 255.0 * (cfg.glcm_levels - 1)).astype(np.uint8)
    glcm = graycomatrix(xq, 
                        distances=cfg.glcm_distances, 
                        angles=cfg.glcm_angles, 
                        levels=cfg.glcm_levels, 
                        symmetric=True, 
                        normed=True)
    feats = {}
    for prop in ("contrast", "homogeneity", "ASM", "correlation"):
        v = graycoprops(glcm, prop)
        feats[prop+"_mean"] = float(np.mean(v))
        feats[prop+"_std"]  = float(np.std(v))
    return feats

# ------------------------- Embeddings & FID ----------------------------------
class FeatureEncoder:
    """Wrap torchvision model to produce pooled embeddings."""
    def __init__(self, device: str = "cpu", arch: str = "resnet18", img_size: int = 224):
        if not TORCH_OK:
            raise RuntimeError("Torch/torchvision not available")
        self.device = torch.device(device)
        self.img_size = img_size
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.out_dim = 512
            self.backbone = torch.nn.Sequential(*(list(m.children())[:-1]))  # global avgpool output
        elif arch == "inception_v3":
            m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            self.out_dim = 2048
            # take feature pool before fc
            self.backbone = torch.nn.Sequential(*(list(m.children())[:-1]))
        else:
            raise ValueError("Unsupported arch")
        self.backbone.eval().to(self.device)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    @torch.no_grad()
    def encode_one(self, x_u8: np.ndarray) -> np.ndarray:
        if x_u8.ndim == 2:
            x3 = np.stack([x_u8]*3, axis=-1)
        else:
            x3 = x_u8
        t = self.tf(x3).unsqueeze(0).to(self.device)
        y = self.backbone(t)
        y = torch.flatten(y, 1)
        return y.cpu().numpy().ravel()

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute Fréchet distance between two Gaussians (FID component)."""
    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    covmean, err = sqrtm(cov_prod, disp=False)
    if err > 1e-6:
        logging.warning("sqrtm warning err=%g; retry with jitter", err)
        jitter = 1e-6 * np.eye(sigma1.shape[0], dtype=sigma1.dtype)
        covmean, _ = sqrtm((sigma1 + jitter) @ (sigma2 + jitter), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

# ------------------------- Tests & distances ---------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size."""
    nx, ny = len(x), len(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    rx = np.sum(ranks[:nx])
    ry = np.sum(ranks[nx:])
    U = rx - nx*(nx+1)/2.0
    delta = (2*U)/(nx*ny) - 1
    return float(delta)

def energy_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy statistic (nonnegative; larger -> more different)."""
    # ||X - Y|| distances
    from sklearn.metrics import pairwise_distances # type: ignore
    dxy = pairwise_distances(X, Y, metric="euclidean")
    dxx = pairwise_distances(X, X, metric="euclidean")
    dyy = pairwise_distances(Y, Y, metric="euclidean")
    nx, ny = X.shape[0], Y.shape[0]
    A = 2.0 * dxy.mean()
    B = dxx[np.triu_indices(nx, 1)].mean() if nx > 1 else 0.0
    C = dyy[np.triu_indices(ny, 1)].mean() if ny > 1 else 0.0
    return float(A - B - C)

def mmd_gaussian(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Biased MMD^2 with Gaussian kernel."""
    from sklearn.metrics.pairwise import rbf_kernel # type: ignore
    Z = np.vstack([X, Y])
    if sigma is None:
        # median heuristic
        from sklearn.metrics import pairwise_distances
        med = np.median(pairwise_distances(Z, Z))
        sigma = med if med > 1e-8 else 1.0
    Kxx = rbf_kernel(X, X, gamma=1.0/(2*sigma**2))
    Kyy = rbf_kernel(Y, Y, gamma=1.0/(2*sigma**2))
    Kxy = rbf_kernel(X, Y, gamma=1.0/(2*sigma**2))
    m = X.shape[0]; n = Y.shape[0]
    mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return float(mmd2)

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of rejections under BH-FDR."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, m+1) / m)
    passed = pvals[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    reject = np.zeros(m, dtype=bool)
    if k > 0:
        reject[pvals <= thresh[k-1]] = True
    return reject

# ------------------------- Visualization -------------------------------------
def save_ecdf_plot(data: Dict[str, np.ndarray], title: str, out: Path):
    plt.figure()
    for label, arr in data.items():
        xs = np.sort(arr)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        plt.plot(xs, ys, label=label)
    plt.xlabel(title)
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_box_plot(data: Dict[str, np.ndarray], title: str, out: Path):
    plt.figure()
    vals = [data[k] for k in data]
    labels = list(data.keys())
    plt.boxplot(vals, tick_labels=labels, showfliers=False)
    plt.ylabel(title)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_qq_plot(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, title: str, out: Path):
    plt.figure()
    percs = np.linspace(1, 99, 99)
    qa = np.percentile(a, percs)
    qb = np.percentile(b, percs)
    plt.scatter(qa, qb, s=8)
    mn = min(qa.min(), qb.min()); mx = max(qa.max(), qb.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(label_a)
    plt.ylabel(label_b)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_heatmap(M: np.ndarray, labels: List[str], title: str, out: Path):
    plt.figure()
    plt.imshow(M, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_power_spectrum_plot(curves: Dict[str, np.ndarray], title: str, out: Path):
    plt.figure()
    for label, y in curves.items():
        x = np.arange(len(y))
        plt.plot(x, y, label=label)
    plt.xlabel("Radial frequency bins")
    plt.ylabel("Mean log power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_metric_panels(metrics: pd.DataFrame, features: List[str], datasets: List[str], out: Path) -> None:
    """Create a multi-row panel where each row has ECDF (left) and boxplot (right) for a feature."""
    features = [f for f in features if f in metrics.columns]
    if not features:
        return
    n_rows = len(features)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 2.2*n_rows), squeeze=False)
    for i, feat in enumerate(features):
        # ECDF
        ax = axes[i, 0]
        for d in datasets:
            vals = metrics.loc[metrics.dataset == d, feat].dropna().values
            if vals.size == 0:
                continue
            xs = np.sort(vals)
            ys = np.linspace(0, 1, len(xs), endpoint=True)
            ax.plot(xs, ys, label=d)
        ax.set_xlabel(feat)
        ax.set_ylabel("ECDF")
        if i == 0:
            ax.legend()
        # Boxplot
        ax = axes[i, 1]
        vals = [metrics.loc[metrics.dataset == d, feat].dropna().values for d in datasets]
        ax.boxplot(vals, tick_labels=datasets, showfliers=False)
        ax.set_ylabel(feat)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def save_qq_panels(metrics: pd.DataFrame, pairs: List[Tuple[str, str]], feats: List[str], out: Path) -> None:
    """Create a panel of QQ plots for selected features and dataset pairs."""
    feats = [f for f in feats if f in metrics.columns]
    if not feats or not pairs:
        return
    n_rows = len(feats)
    n_cols = len(pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 2.4*n_rows), squeeze=False)
    for r, feat in enumerate(feats):
        for c, (A, B) in enumerate(pairs):
            ax = axes[r, c]
            a = metrics.loc[metrics.dataset == A, feat].dropna().values
            b = metrics.loc[metrics.dataset == B, feat].dropna().values
            if a.size == 0 or b.size == 0:
                continue
            percs = np.linspace(1, 99, 99)
            qa = np.percentile(a, percs)
            qb = np.percentile(b, percs)
            ax.scatter(qa, qb, s=6)
            mn = min(qa.min(), qb.min()); mx = max(qa.max(), qb.max())
            ax.plot([mn, mx], [mn, mx])
            ax.set_xlabel(f"{A} {feat}")
            ax.set_ylabel(f"{B} {feat}")
            if r == 0:
                ax.set_title(f"QQ: {A} vs {B}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def save_umap_plot(embeds: pd.DataFrame, out: Path, label_col: str = "dataset") -> None:
    """2D UMAP of deep embeddings with dataset coloring."""
    if embeds.empty:
        logging.info("UMAP skipped: empty embeddings.")
        return
    feat_cols = [c for c in embeds.columns if c.startswith("f")]
    if not feat_cols:
        logging.info("UMAP skipped: no feature columns found.")
        return
    if not UMAP_OK:
        logging.warning("umap-learn not available. Skipping UMAP plot.")
        return
    X = embeds[feat_cols].values
    y = embeds[label_col].values
    Xs = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=17)
    Z = reducer.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = np.unique(y)
    for cls in classes:
        mask = (y == cls)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, label=str(cls))
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

# ------------------------- Main pipeline -------------------------------------
def analyze_dataset(name: str, files: List[Path], cfg: Config, enc: Optional[FeatureEncoder]) -> Dict[str, Any]:
    rows = []
    glcm_rows = []
    embeds = []
    power_curves: list[np.ndarray] = []
    rng = np.random.default_rng(cfg.random_state)
    if cfg.max_images_per_set:
        files = files[:cfg.max_images_per_set]
    for p in tqdm(files, desc=f"Processing {name}"):
        try:
            x = imread_uint8_gray(p)
            h, w = x.shape
            row = {
                "dataset": name, "path": str(p), "width": int(w), "height": int(h),
                "aspect": float(w/h)
            }
            row.update(basic_stats(x))
            row.update(gradient_edge_stats(x))
            ps = radial_power_spectrum_slope(x)
            row.update(ps)
            rows.append(row)
            glcm_rows.append({"dataset": name, "path": str(p), **glcm_features(x, cfg)})
            # store log power curve for a representative subset
            if len(power_curves) < 128:
                F = fftshift(fft2((x.astype(np.float32)/255.0) - np.mean(x)/255.0))
                P = np.abs(F)**2
                h2, w2 = x.shape
                cy, cx = h2//2, w2//2
                yv, xv = np.ogrid[:h2, :w2]
                r = np.hypot(yv - cy, xv - cx).astype(np.int32)
                maxr = min(cy, cx)
                radial_mean = np.bincount(r.ravel(), P.ravel())[:maxr] / np.maximum(1, np.bincount(r.ravel())[:maxr])
                ylog = np.log(radial_mean[1:] + 1e-12)
                power_curves.append(ylog)
            if enc is not None:
                emb = enc.encode_one(x)
                embeds.append({"dataset": name, "path": str(p), **{f"f{i}": float(v) for i, v in enumerate(emb)}})
        except Exception as e:
            logging.warning("Failed on %s: %s", p, e)
    return {
        "metrics": pd.DataFrame(rows),
        "glcm": pd.DataFrame(glcm_rows),
        "embeddings": pd.DataFrame(embeds) if len(embeds) else pd.DataFrame(),
        "power_curves": np.array(power_curves, dtype=np.float32)
    }

def pairwise_dataset_tests(df: pd.DataFrame, feature_cols: List[str], label_col: str = "dataset") -> pd.DataFrame:
    """Run KS, MWU, Cliff's δ for each scalar feature across dataset pairs."""
    datasets = df[label_col].unique().tolist()
    rows = []
    for feat in feature_cols:
        groups = {d: df.loc[df[label_col]==d, feat].dropna().values for d in datasets}
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                a, b = groups[datasets[i]], groups[datasets[j]]
                if len(a) < 5 or len(b) < 5:
                    continue
                ks = stats.ks_2samp(a, b, method="asymp")
                mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
                delta = cliffs_delta(a, b)
                rows.append({
                    "feature": feat, "A": datasets[i], "B": datasets[j],
                    "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                    "mwu_u": float(mwu.statistic), "mwu_p": float(mwu.pvalue),
                    "cliffs_delta": float(delta),
                    "A_mean": float(np.mean(a)), "B_mean": float(np.mean(b)),
                    "A_median": float(np.median(a)), "B_median": float(np.median(b))
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["reject_fdr_ks"]  = benjamini_hochberg(out["ks_p"].values, alpha=0.05)
        out["reject_fdr_mwu"] = benjamini_hochberg(out["mwu_p"].values, alpha=0.05)
    return out

def multivariate_tests(df: pd.DataFrame, cols: List[str], label_col: str = "dataset") -> pd.DataFrame:
    """Energy statistic and MMD on multivariate vectors."""
    datasets = df[label_col].unique().tolist()
    rows = []
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            A = df.loc[df[label_col]==datasets[i], cols].dropna().values
            B = df.loc[df[label_col]==datasets[j], cols].dropna().values
            if len(A) < 5 or len(B) < 5:
                continue
            en = energy_statistic(A, B)
            mmd2 = mmd_gaussian(A, B, sigma=None)
            rows.append({"features": "|".join(cols), "A": datasets[i], "B": datasets[j],
                         "energy": en, "mmd2": mmd2})
    return pd.DataFrame(rows)

def compute_fid_between(emb_df: pd.DataFrame, label_col: str = "dataset", path_cols_prefix: str = "f") -> pd.DataFrame:
    """Assumes embeddings are from a single encoder; computes FID between dataset pairs by Gaussian fit."""
    datasets = emb_df[label_col].unique().tolist()
    D = [c for c in emb_df.columns if c.startswith(path_cols_prefix)]
    rows = []
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            A = emb_df.loc[emb_df[label_col]==datasets[i], D].values
            B = emb_df.loc[emb_df[label_col]==datasets[j], D].values
            muA, muB = A.mean(0), B.mean(0)
            sigA = np.cov(A, rowvar=False)
            sigB = np.cov(B, rowvar=False)
            fid = frechet_distance(muA, sigA, muB, sigB)
            rows.append({"A": datasets[i], "B": datasets[j], "fid": float(fid)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_dir", required=True, type=Path)
    ap.add_argument("--seg_dir", required=True, type=Path)
    ap.add_argument("--retina_seg_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_images_per_set", type=int, default=None)
    ap.add_argument("--compute_embeddings", type=str, default="true")
    ap.add_argument("--compute_fid", type=str, default="true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    # Configure logging (compatible with tqdm)
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove default handlers if re-running in notebook context
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    h = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
    h.setFormatter(formatter)
    root_logger.addHandler(h)
    cfg = Config(det_dir=args.det_dir, seg_dir=args.seg_dir, retina_seg_dir=args.retina_seg_dir,
                 out_dir=args.out_dir, max_images_per_set=args.max_images_per_set,
                 compute_embeddings=(args.compute_embeddings.lower()=="true"),
                 compute_fid=(args.compute_fid.lower()=="true"))

    # Configure plotting style
    configure_matplotlib()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "figs").mkdir(exist_ok=True)

    # Early skip logic: if core CSVs exist, skip heavy recomputation
    metrics_path = cfg.out_dir / "metrics.csv"
    glcm_path = cfg.out_dir / "glcm_features.csv"
    embeds_path = cfg.out_dir / "embeddings.csv"
    skip_analysis = False
    if metrics_path.exists() and glcm_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            glcm_df = pd.read_csv(glcm_path)
            embeds = pd.read_csv(embeds_path) if embeds_path.exists() else pd.DataFrame()
            skip_analysis = True
            logging.info("Found existing metrics/glcm CSVs. Skipping recomputation; generating plots only.")
        except Exception as e:
            logging.warning("Could not read existing CSVs, recomputing. Error: %s", e)
            skip_analysis = False
    if not skip_analysis:
        det_files = list_images(cfg.det_dir, cfg.image_exts)
        seg_files = list_images(cfg.seg_dir, cfg.image_exts)
        ret_files = list_images(cfg.retina_seg_dir, cfg.image_exts)
        logging.info("Found images: det=%d seg=%d retina_seg=%d", len(det_files), len(seg_files), len(ret_files))

        enc = None
        if cfg.compute_embeddings and TORCH_OK:
            logging.info("Building encoder for embeddings (ResNet18 @ %dx%d)", cfg.embed_image_size, cfg.embed_image_size)
            enc = FeatureEncoder(device=args.device, arch="resnet18", img_size=cfg.embed_image_size)
        elif cfg.compute_embeddings and not TORCH_OK:
            logging.warning("Torch/torchvision missing. Skipping embeddings.")

        A = analyze_dataset("angio_det", det_files, cfg, enc)
        B = analyze_dataset("angio_seg", seg_files, cfg, enc)
        C = analyze_dataset("retina_seg", ret_files, cfg, enc)
        metrics = pd.concat([A["metrics"], B["metrics"], C["metrics"]], ignore_index=True)
        glcm_df = pd.concat([A["glcm"], B["glcm"], C["glcm"]], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)
        glcm_df.to_csv(glcm_path, index=False)
        logging.info("Saved metrics/glcm CSVs. metrics=%s glcm=%s", metrics_path, glcm_path)
        embeds = pd.DataFrame()
        if cfg.compute_embeddings and enc is not None:
            list_embeds = [A["embeddings"], B["embeddings"], C["embeddings"]]
            list_embeds = [e for e in list_embeds if not e.empty]
            if list_embeds:
                embeds = pd.concat(list_embeds, ignore_index=True)
                embeds.to_csv(embeds_path, index=False)
                logging.info("Saved embeddings CSV. path=%s shape=%s", embeds_path, embeds.shape)
        # Run statistical tests only when recomputing
        scalar_cols = ["width","height","aspect","mean","std","skew","kurtosis","entropy",
                       "sobel_p50","sobel_p90","lap_var","edge_density","ps_slope","ps_intercept"]
        scalar_cols = [c for c in scalar_cols if c in metrics.columns]
        tests_scalar = pairwise_dataset_tests(metrics, scalar_cols, "dataset")
        tests_scalar.to_csv(cfg.out_dir / "stats_scalar_tests.csv", index=False)
        tex_cols = [c for c in glcm_df.columns if c not in ("dataset","path")]
        tests_texture = multivariate_tests(glcm_df, tex_cols, "dataset")
        tests_texture.to_csv(cfg.out_dir / "stats_texture_tests.csv", index=False)
        compact_cols = ["mean","std","lap_var"]
        tests_compact = multivariate_tests(metrics, compact_cols, "dataset")
        tests_compact.to_csv(cfg.out_dir / "stats_compact_tests.csv", index=False)
        fid_df = pd.DataFrame()
        if cfg.compute_embeddings and cfg.compute_fid and (not embeds.empty):
            fid_df = compute_fid_between(embeds, label_col="dataset", path_cols_prefix="f")
            fid_path = cfg.out_dir / "fid_pairs.csv"
            fid_df.to_csv(fid_path, index=False)
            logging.info("Computed FID for %d pairs -> %s", len(fid_df), fid_path)
    else:
        # When skipping, define scalar_cols for plotting reuse
        scalar_cols = [c for c in ["width","height","aspect","mean","std","skew","kurtosis","entropy",
                                   "sobel_p50","sobel_p90","lap_var","edge_density","ps_slope","ps_intercept"] if c in metrics.columns]

    # Visualizations
    dsets = ["angio_det","angio_seg","retina_seg"]
    # Combined ECDF+Box panels for key features
    key_feats = [f for f in ["mean","std","lap_var","edge_density","ps_slope"] if f in scalar_cols]
    if key_feats:
        save_metric_panels(metrics, key_feats, dsets, cfg.out_dir / "figs" / "panels_metrics.png")
    # Also keep per-feature single figures for granular inspection
    for feat in key_feats:
        data = {d: metrics.loc[metrics.dataset==d, feat].values for d in dsets}
        save_ecdf_plot(data, feat, cfg.out_dir / "figs" / f"ecdf_{feat}.png")
        save_box_plot(data, feat, cfg.out_dir / "figs" / f"box_{feat}.png")

    # QQ plots for selected pairs
    qq_feats = [f for f in ["mean","lap_var"] if f in scalar_cols]
    qq_pairs: List[Tuple[str, str]] = [("retina_seg","angio_seg"), ("angio_det","angio_seg")]
    if qq_feats:
        save_qq_panels(metrics, qq_pairs, qq_feats, cfg.out_dir / "figs" / "qq_panels.png")
        # Keep the original individual QQs for quick look
        if "mean" in qq_feats:
            a = metrics.loc[metrics.dataset=="retina_seg","mean"].values
            b = metrics.loc[metrics.dataset=="angio_seg","mean"].values
            save_qq_plot(a,b,"retina_seg","angio_seg","QQ mean", cfg.out_dir / "figs" / "qq_mean_retina_vs_angioseg.png")
        if "lap_var" in qq_feats:
            a = metrics.loc[metrics.dataset=="angio_det","lap_var"].values
            b = metrics.loc[metrics.dataset=="angio_seg","lap_var"].values
            save_qq_plot(a,b,"angio_det","angio_seg","QQ lap_var", cfg.out_dir / "figs" / "qq_lapvar_det_vs_seg.png")

    # Power spectrum curves
    def mean_log_curve(curves: np.ndarray) -> np.ndarray:
        if curves.size == 0:
            return np.array([])
        L = min([len(c) for c in curves])
        return curves[:, :L].mean(0)

    if not skip_analysis:
        curves = {
            "angio_det": mean_log_curve(A["power_curves"]),
            "angio_seg": mean_log_curve(B["power_curves"]),
            "retina_seg": mean_log_curve(C["power_curves"]),
        }
        curves = {k:v for k,v in curves.items() if v.size}
        if curves:
            save_power_spectrum_plot(curves, "Mean log radial power", cfg.out_dir / "figs" / "power_spectra.png")
    else:
        logging.info("Skipping power spectrum plots (raw FFT curves unavailable without recomputation).")

    # Embedding geometry: pairwise dataset centroid distances heatmap
    if not embeds.empty:
        groups = {d: embeds.loc[embeds.dataset==d, [c for c in embeds.columns if c.startswith("f")]].values for d in dsets if d in embeds.dataset.unique()}
        if groups:
            cent = {d: groups[d].mean(0) for d in groups}
            lab = list(cent.keys())
            M = np.zeros((len(lab), len(lab)), dtype=float)
            for i, a in enumerate(lab):
                for j, b in enumerate(lab):
                    M[i,j] = float(np.linalg.norm(cent[a]-cent[b]))
            save_heatmap(M, lab, "Centroid distances in embedding space", cfg.out_dir / "figs" / "embedding_centroid_distances.png")
            # UMAP projection of embeddings
            save_umap_plot(embeds, cfg.out_dir / "figs" / "umap_embeddings.png")

    # Aggregate report
    if not skip_analysis:
        report_rows = []
        for fname, tag in [("stats_scalar_tests.csv","scalar"), ("stats_texture_tests.csv","texture"), ("stats_compact_tests.csv","compact")]:
            fpath = cfg.out_dir / fname
            if fpath.exists():
                df_ = pd.read_csv(fpath)
                if not df_.empty:
                    df_["kind"] = tag
                    report_rows.append(df_)
        report = pd.concat(report_rows, ignore_index=True) if report_rows else pd.DataFrame()
        report.to_csv(cfg.out_dir / "stats_report.csv", index=False)
    else:
        logging.info("Skipping stats_report aggregation (analysis skipped).")
    logging.info("Done. Outputs at: %s", cfg.out_dir)

if __name__ == "__main__":
    main()
