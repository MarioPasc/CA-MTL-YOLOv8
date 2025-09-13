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

def fov_mask_from_raw(gray:np.ndarray)->np.ndarray:
    """
    Binary FOV from raw grayscale:
      Gaussian blur → Otsu → keep largest component → morphological close → return bool
    """
    g = cv2.GaussianBlur(gray, (0,0), 3.0)
    _,thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m1 = thr>0; m2 = ~m1
    mask = m1 if np.count_nonzero(m1)>=np.count_nonzero(m2) else m2
    mask = mask.astype(np.uint8)*255
    num, labels = cv2.connectedComponents(mask)
    if num>1:
        counts = np.bincount(labels.ravel()); counts[0]=0
        keep = counts.argmax()
        mask = (labels==keep).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    return mask>0

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

# -------- Orchestrator --------

def preprocess(img:np.ndarray, cfg:Cfg, log:logging.Logger)->np.ndarray:
    g0 = to_gray(img, cfg.use_green)
    fov = fov_mask_from_raw(g0)

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
