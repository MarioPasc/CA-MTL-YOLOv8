#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fundus preprocessing with optional Foracchia-style normalization + preserved black background → uint8 grayscale.

Pipeline (when --normalize foracchia):
  1) Grayscale (green channel by default).
  2) FOV mask from raw grayscale; background kept black at the end.
  3) Foracchia-style luminosity & contrast normalization inside FOV:
       L(x,y) = masked Gaussian mean at scale σ_L
       C(x,y) = masked Gaussian std  at scale σ_C
       Z = (I - L)/C  → affine retarget to target_mean/target_std inside FOV
  4) Optional mild CLAHE, denoise, gamma.
  5) Re-apply FOV mask; cast to uint8.

References:
- Foracchia, Grisan, Ruggeri. “Luminosity and contrast normalization in retinal images.” *Medical Image Analysis*, 2005. :contentReference[oaicite:0]{index=0}
- DRIVE provides FOV masks and motivates masking background in fundus processing. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
import argparse, glob, logging, os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Cfg:
    clip_limit: float = 1.5
    tile_grid: int = 32
    shade_ksize: int = 128
    median_ksize: int = 2
    gamma: float = 1.0
    resize_to: Optional[Tuple[int,int]] = None
    use_green: bool = True
    # Normalization controls
    normalize: str = "foracchia"        # {"none","foracchia"}
    sigma_lum: float = 10.0         # Foracchia: luminosity scale (pixels)
    sigma_con: float = 64.0         # Foracchia: contrast scale (pixels)
    target_mean: float = 128.0      # Foracchia: target mean within FOV
    target_std: float = 18.0        # Foracchia: target std within FOV

class PreprocessError(Exception): pass

def _ensure_odd(k:int)->int: k=max(1,int(k)); return k if k%2==1 else k+1
def _to_uint8(x:np.ndarray)->np.ndarray: return np.clip(x,0,255).astype(np.uint8, copy=False)

def read_image(path:str)->np.ndarray:
    img=cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise PreprocessError(f"Cannot read {path}")
    return img

def to_gray(img:np.ndarray, use_green:bool)->np.ndarray:
    if img.ndim==2: g=img
    elif img.ndim==3 and img.shape[2]>=3: g=img[:,:,1] if use_green else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: raise PreprocessError("Unsupported image")
    if g.dtype!=np.uint8:
        mn, mx = float(g.min()), float(g.max())
        g = np.zeros_like(g, dtype=np.uint8) if mx<=mn else _to_uint8(255*(g-mn)/(mx-mn))
    return g

# -------- Foracchia-style normalization --------

def _masked_gauss(img_f: np.ndarray, mask_f: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized masked Gaussian smoothing: conv(img*mask)/conv(mask)."""
    k = int(6*sigma+1) | 1
    num = cv2.GaussianBlur(img_f*mask_f, (k,k), sigma)
    den = cv2.GaussianBlur(mask_f, (k,k), sigma)
    return num / (den + 1e-6)

def foracchia_normalize(gray: np.ndarray,
                        fov: np.ndarray,
                        sigma_lum: float,
                        sigma_con: float,
                        target_mean: float,
                        target_std: float) -> np.ndarray:
    """
    Foracchia et al. normalization:
      L = masked Gaussian mean at σ_L;  C from masked Gaussian variance at σ_C.
      Z = (I - L)/C; affine retarget within FOV to target_mean/target_std.
    """
    g = gray.astype(np.float32)
    m = fov.astype(np.float32)

    L  = _masked_gauss(g, m, sigma_lum)
    mu = _masked_gauss(g, m, sigma_con)
    mu2= _masked_gauss(g*g, m, sigma_con)
    var= np.maximum(mu2 - mu*mu, 1e-6)
    C  = np.sqrt(var)

    Z = (g - L) / C
    if np.any(fov):
        z_m = float(Z[fov].mean())
        z_s = float(Z[fov].std()) if Z[fov].size else 1.0
    else:
        z_m, z_s = 0.0, 1.0

    out = (Z - z_m) * (target_std / max(z_s, 1e-6)) + target_mean
    out[~fov] = 0.0
    return _to_uint8(out)

# -------- Legacy steps --------

def shade_correction(gray:np.ndarray, ksize:int)->np.ndarray:
    k=_ensure_odd(ksize)
    bg = cv2.medianBlur(gray, k)
    return _to_uint8(gray.astype(np.int16)-bg.astype(np.int16)+128)

def clahe(gray:np.ndarray, clip:float, tile:int)->np.ndarray:
    cla = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(max(2,int(tile)),)*2)
    return cla.apply(gray)

def denoise_median(gray:np.ndarray, k:int)->np.ndarray:
    k=_ensure_odd(k)
    return gray if k<3 else cv2.medianBlur(gray, k)

def gamma_corr(gray:np.ndarray, gamma:float)->np.ndarray:
    if abs(gamma-1.0)<1e-6: return gray
    inv=1.0/float(gamma)
    lut = ((np.arange(256)/255.0)**inv*255.0).astype(np.uint8)
    return cv2.LUT(gray,lut)

def fov_mask_from_raw(img: np.ndarray,
                      border_bg: int = 25,
                      erode_fg: int = 35,
                      iters: int = 5,
                      log: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Robust FOV via seeded GrabCut + optional ellipse regularization.

    Seeds:
      - Background: a 'border_bg' wide frame and the dark corners.
      - Foreground: eroded coarse fundus from saturation ∪ red channel.

    Steps:
      1) Build coarse mask: Otsu on blurred Saturation and Red; union + largest CC.
      2) Create GrabCut label map: definite BG on borders, definite FG on eroded coarse mask.
      3) Run GrabCut (INIT_WITH_MASK).
      4) Clean: largest CC, fill small holes, close.
      5) Fit ellipse to contour; intersect with GrabCut mask only if it *expands* locally.

    Returns bool mask (True = inside FOV).
    References: GrabCut graph-cut segmentation. OpenCV ellipse fitting. Foracchia normalization. 
    """
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img
    else:
        bgr = img
        gray = bgr[:, :, 1]

    H, W = gray.shape
    # --- coarse mask from color cues ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = cv2.GaussianBlur(hsv[:, :, 1], (0, 0), 5)
    R = cv2.GaussianBlur(bgr[:, :, 2], (0, 0), 5)
    _, thS = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thR = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coarse = cv2.morphologyEx(cv2.bitwise_or(thS, thR), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    # keep largest CC
    num, lab = cv2.connectedComponents(coarse)
    if num > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        coarse = (lab == cnt.argmax()).astype(np.uint8) * 255

    # --- GrabCut seeds ---
    mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    # definite background frame
    mask[:border_bg, :] = cv2.GC_BGD
    mask[-border_bg:, :] = cv2.GC_BGD
    mask[:, :border_bg] = cv2.GC_BGD
    mask[:, -border_bg:] = cv2.GC_BGD
    # corners as BG
    rr = border_bg
    mask[:rr, :rr] = cv2.GC_BGD; mask[:rr, -rr:] = cv2.GC_BGD
    mask[-rr:, :rr] = cv2.GC_BGD; mask[-rr:, -rr:] = cv2.GC_BGD
    # definite foreground from eroded coarse mask
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, erode_fg)|1,)*2)
    fg_seed = cv2.erode(coarse, er)
    mask[fg_seed > 0] = cv2.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    gc = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8) * 255

    # --- clean up ---
    gc = cv2.morphologyEx(gc, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    num, lab = cv2.connectedComponents(gc)
    if num > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        gc = (lab == cnt.argmax()).astype(np.uint8) * 255
    # fill holes
    inv = cv2.bitwise_not(gc)
    num, lab = cv2.connectedComponents(inv)
    for k in range(1, num):
        hole = (lab == k)
        if hole.sum() < 0.02 * (gc > 0).sum():
            gc[hole] = 255

    fov_gc = gc > 0

    # --- optional ellipse regularization ---
    cnts, _ = cv2.findContours(gc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts and len(cnts[0]) >= 5:
        (cx, cy), (MA, ma), ang = cv2.fitEllipseAMS(cnts[0])  # robust variant
        ell = np.zeros_like(gc)
        cv2.ellipse(ell, ((cx, cy), (MA, ma), ang), 255, -1)
        ell = ell > 0
        # intersect only where ellipse expands mask by <= 2% area
        if ell.sum() <= 1.02 * fov_gc.sum():
            fov = ell & (cv2.dilate(fov_gc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        else:
            fov = fov_gc
    else:
        fov = fov_gc

    if log:
        log.debug("GrabCut FOV: area=%.3f, seeds fg=%d, params border_bg=%d erode_fg=%d",
                  fov.mean(), int((fg_seed > 0).sum()), border_bg, erode_fg)
    return fov

def _largest_cc_bool(bw: np.ndarray) -> np.ndarray:
    num, lab = cv2.connectedComponents(bw.astype(np.uint8))
    if num <= 1: 
        return bw.astype(bool)
    cnt = np.bincount(lab.ravel()); cnt[0] = 0
    return (lab == cnt.argmax())

def fov_mask_fallback_strict(img: np.ndarray) -> np.ndarray:
    """
    Stricter, color-aware fallback:
      - Use red-band + triangle threshold (more conservative than Otsu on bright lesions),
      - Union with saturation-threshold mask,
      - Keep largest CC, close, fill tiny holes.
    """
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
    R = cv2.GaussianBlur(bgr[:, :, 2], (0, 0), 7)
    S = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1], (0, 0), 7)

    # Triangle threshold is conservative for unimodal backgrounds
    _, thR = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    _, thS = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.bitwise_or(thR, thS)

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    m = _largest_cc_bool(m > 0)
    # fill small holes
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv)
    area = m.sum()
    for k in range(1, n):
        hole = (lab == k)
        if hole.sum() < 0.01 * area:
            m[hole] = True
    return m

# -------- Orchestrator --------

def preprocess_with_mask(img: np.ndarray, cfg: Cfg, log: logging.Logger
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as preprocess(), but also returns the boolean FOV mask used.
    On failure patterns (implausible area), retries with a stricter fallback.
    """
    g0 = to_gray(img, cfg.use_green)
    fov = fov_mask_from_raw(img=g0, log=log)

    # sanity: FOV area must be in [0.30, 0.80]
    a = fov.mean()
    if not (0.30 <= a <= 0.80):
        if log: log.debug(f"[fundus] FOV area {a:.3f} out of range; retrying with strict fallback")
        fov = fov_mask_fallback_strict(img)

    # the remainder mirrors preprocess()
    if cfg.normalize.lower() == "foracchia":
        g = foracchia_normalize(
            g0, fov,
            sigma_lum=cfg.sigma_lum,
            sigma_con=cfg.sigma_con,
            target_mean=cfg.target_mean,
            target_std=cfg.target_std
        )
        if cfg.clip_limit > 0:
            g = clahe(g, cfg.clip_limit, cfg.tile_grid)
    else:
        g = shade_correction(g0, cfg.shade_ksize)
        g = clahe(g, cfg.clip_limit, cfg.tile_grid)

    g = denoise_median(g, cfg.median_ksize)
    g = gamma_corr(g, cfg.gamma)

    if cfg.resize_to:
        g   = cv2.resize(g, cfg.resize_to, interpolation=cv2.INTER_AREA)
        fov = cv2.resize(fov.astype(np.uint8)*255, cfg.resize_to,
                         interpolation=cv2.INTER_NEAREST) > 0

    g[~fov] = 0
    return _to_uint8(g), fov

def preprocess(img:np.ndarray, cfg:Cfg, log:logging.Logger)->np.ndarray:
    g0 = to_gray(img, cfg.use_green)
    fov = fov_mask_from_raw(img=g0, log=log)
    
    if cfg.normalize.lower() == "foracchia":
        g = foracchia_normalize(
            g0, fov,
            sigma_lum=cfg.sigma_lum,
            sigma_con=cfg.sigma_con,
            target_mean=cfg.target_mean,
            target_std=cfg.target_std
        )
        # Optional mild CLAHE after normalization
        if cfg.clip_limit > 0:
            g = clahe(g, cfg.clip_limit, cfg.tile_grid)
    else:
        # Legacy path
        g  = shade_correction(g0, cfg.shade_ksize)
        g  = clahe(g, cfg.clip_limit, cfg.tile_grid)

    g  = denoise_median(g, cfg.median_ksize)
    g  = gamma_corr(g, cfg.gamma)

    if cfg.resize_to:
        g   = cv2.resize(g, cfg.resize_to, interpolation=cv2.INTER_AREA)
        fov = cv2.resize(fov.astype(np.uint8)*255, cfg.resize_to, interpolation=cv2.INTER_NEAREST)>0

    g[~fov] = 0
    return _to_uint8(g)

# ---------- QC + retry helpers ----------

def mask_metrics(fov: np.ndarray) -> tuple[float, float, float]:
    """Return (area_frac, compactness, hole_ratio) for a boolean FOV."""
    h, w = fov.shape
    A = float(np.count_nonzero(fov))
    area_frac = A / float(h*w)
    if A <= 0:
        return 0.0, 0.0, 1.0
    cnts, _ = cv2.findContours(fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = cv2.arcLength(cnts[0], True) if cnts else 1.0
    compact = float(4.0*np.pi*A / max(per*per, 1.0))
    # holes
    inv = (~fov).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv)
    hole_area = 0
    for k in range(1, n):
        hole = (lab == k)
        # count only holes fully inside
        if 0 < hole.sum() < (h*w) and not (hole[0,:].any() or hole[-1,:].any() or hole[:,0].any() or hole[:,-1].any()):
            hole_area += int(hole.sum())
    hole_ratio = float(hole_area) / max(A, 1.0)
    return area_frac, max(0.0, min(1.0, compact)), hole_ratio

def _iter_inputs(inp:str):
    if os.path.isdir(inp):
        exts=("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
        for e in exts: yield from glob.glob(os.path.join(inp,e))
    else:
        paths = glob.glob(inp) or ([inp] if os.path.isfile(inp) else [])
        if not paths: raise PreprocessError(f"No inputs match {inp}")
        for p in paths: yield p

def main():
    ap=argparse.ArgumentParser(description="Fundus preprocessing with optional Foracchia normalization and preserved black background.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--shade-ksize", type=int, default=61)
    ap.add_argument("--median-ksize", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W","H"))
    ap.add_argument("--normalize", type=str, choices=["none","foracchia"], default="none",
                    help="Apply luminosity/contrast normalization before enhancement.")
    ap.add_argument("--sigma-lum", type=float, default=30.0)
    ap.add_argument("--sigma-con", type=float, default=30.0)
    ap.add_argument("--tmean", type=float, default=128.0)
    ap.add_argument("--tstd", type=float, default=40.0)
    ap.add_argument("-v","--verbose", action="store_true")
    args=ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log=logging.getLogger("fundus")
    os.makedirs(args.outdir, exist_ok=True)
    cfg=Cfg(
        clip_limit=args.clip_limit,
        tile_grid=args.tile,
        shade_ksize=args.shade_ksize,
        median_ksize=args.median_ksize,
        gamma=args.gamma,
        resize_to=tuple(args.resize) if args.resize else None,
        use_green=True,
        normalize=args.normalize,
        sigma_lum=args.sigma_lum,
        sigma_con=args.sigma_con,
        target_mean=args.tmean,
        target_std=args.tstd
    )
    for p in _iter_inputs(args.input):
        img=read_image(p); out=preprocess(img,cfg,log)
        base=os.path.splitext(os.path.basename(p))[0]
        op=os.path.join(args.outdir,f"{base}_proc.png")
        if not cv2.imwrite(op,out): raise PreprocessError(f"Write failed {op}")
        log.info(f"Wrote {op} shape={out.shape} dtype={out.dtype}")
if __name__=="__main__": main()
