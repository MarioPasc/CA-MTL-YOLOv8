#!/usr/bin/env python3
"""
Preprocessing for grayscale coronary X-ray angiography (XCA) → uint8.

Design goals:
  • Correct spatially varying illumination (flat-field / homomorphic). 
  • Improve local contrast cautiously; avoid noise blow-up common with CLAHE on X-ray. 
  • Preserve vessel edges; optional vesselness enhancement for downstream tasks.

References:
  • Software-based illumination correction improves quantitative assays (flat-field). 
  • Homomorphic filtering removes uneven illumination and boosts contrast. 
  • In X-ray/ultrasound, aggressive CLAHE can amplify noise; use mild settings. 
  • Vesselness filters (Frangi/Sato) are effective for angiography vessel enhancement.
"""
from __future__ import annotations
import argparse, logging, os, glob
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2, numpy as np

try:
    from skimage.filters import frangi  # optional vesselness
    _HAS_SKIMAGE=True
except Exception:
    _HAS_SKIMAGE=False

@dataclass(frozen=True)
class CfgXCA:
    homomorphic_sigma: float = 30.0    # std of Gaussian LPF in pixels (illumination scale)
    clahe_clip: float = 1.5            # mild CLAHE; set 0 to disable
    clahe_tile: int = 8
    denoise_h: float = 7.0             # fastNlMeansDenoising strength
    unsharp_amount: float = 0.5        # 0 disables; positive sharpens
    vesselness: bool = False           # use Frangi/Sato if available
    resize_to: Optional[Tuple[int,int]] = None

class PreXCAError(Exception): pass

def _to_uint8(x:np.ndarray)->np.ndarray: return np.clip(x,0,255).astype(np.uint8, copy=False)

def read_gray(path:str)->np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise PreXCAError(f"Cannot read {path}")
    if img.ndim==3: img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype!=np.uint8:
        mn, mx = float(img.min()), float(img.max())
        img = np.zeros_like(img, dtype=np.uint8) if mx<=mn else _to_uint8(255*(img-mn)/(mx-mn))
    return img

def homomorphic(gray:np.ndarray, sigma:float)->np.ndarray:
    """
    Log-domain homomorphic filtering:
      log(I+1) → subtract low-pass (illumination) via Gaussian blur → exp → rescale
    Removes multiplicative shading typical in X-ray fluoroscopy.
    """
    g = gray.astype(np.float32)/255.0
    logi = np.log1p(g)
    lp = cv2.GaussianBlur(logi, (0,0), sigma)
    hp = logi - lp
    out = np.expm1(hp)
    out = (out - out.min()) / max(1e-6, (out.max()-out.min()))
    return _to_uint8(out*255.0)

def clahe(gray:np.ndarray, clip:float, tile:int)->np.ndarray:
    if clip<=0: return gray
    cla = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(max(2,int(tile)),)*2)
    return cla.apply(gray)

def denoise(gray:np.ndarray, h:float)->np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=float(h), templateWindowSize=7, searchWindowSize=21)

def unsharp(gray:np.ndarray, amount:float)->np.ndarray:
    if amount<=0: return gray
    blur = cv2.GaussianBlur(gray,(0,0),1.0)
    return _to_uint8(cv2.addWeighted(gray, 1.0+amount, blur, -amount, 0))

def enhance_vesselness(gray:np.ndarray)->np.ndarray:
    if not _HAS_SKIMAGE: return gray
    # Frangi responds to tubular structures; result in [0,1]
    v = frangi(gray.astype(np.float32)/255.0, sigmas=(1,2,3), alpha=0.5, beta=0.5, gamma=15)
    v = (v - np.nanmin(v)) / max(1e-6, (np.nanmax(v)-np.nanmin(v)))
    return _to_uint8(v*255.0)

def preprocess_xca(g:np.ndarray, cfg:CfgXCA)->np.ndarray:
    g = homomorphic(g, cfg.homomorphic_sigma)          # illumination correction
    g = clahe(g, cfg.clahe_clip, cfg.clahe_tile)       # mild local contrast
    g = denoise(g, cfg.denoise_h)                      # edge-preserving denoise
    g = unsharp(g, cfg.unsharp_amount)                 # small detail boost
    if cfg.vesselness:
        v = enhance_vesselness(g)
        # Blend vesselness as guidance to avoid hallucinations
        g = _to_uint8(0.7*g + 0.3*v)
    if cfg.resize_to:
        g = cv2.resize(g, cfg.resize_to, interpolation=cv2.INTER_AREA)
    return _to_uint8(g)

def main():
    ap=argparse.ArgumentParser(description="Grayscale coronary angiography preprocessing → uint8.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--homo-sigma", type=float, default=30.0)
    ap.add_argument("--clahe-clip", type=float, default=1.5)
    ap.add_argument("--clahe-tile", type=int, default=8)
    ap.add_argument("--denoise-h", type=float, default=7.0)
    ap.add_argument("--unsharp", type=float, default=0.5)
    ap.add_argument("--vesselness", action="store_true")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W","H"))
    ap.add_argument("-v","--verbose", action="store_true")
    args=ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    os.makedirs(args.outdir, exist_ok=True)

    cfg=CfgXCA(args.homo_sigma, args.clahe_clip, args.clahe_tile,
               args.denoise_h, args.unsharp, args.vesselness,
               tuple(args.resize) if args.resize else None)

    paths = glob.glob(args.input) or ([args.input] if os.path.isfile(args.input) else [])
    if not paths: raise PreXCAError(f"No inputs match {args.input}")
    for p in paths:
        g = read_gray(p)
        out = preprocess_xca(g, cfg)
        base=os.path.splitext(os.path.basename(p))[0]
        op=os.path.join(args.outdir, f"{base}_xca.png")
        if not cv2.imwrite(op, out): raise PreXCAError(f"Write failed {op}")

if __name__=="__main__": main()
