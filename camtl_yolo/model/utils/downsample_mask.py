import os, math, cv2, numpy as np


def downsample_mask_prob(mask: np.ndarray, stride: int, method: str = "avgpool") -> np.ndarray:
    """
    Downsample a binary mask to a probability mask in [0,1] by block average.
    - 'area' -> cv2.INTER_AREA (equiv. promedio espacial)
    - 'avgpool' -> promedio por bloques exacto cuando stride divide H,W
    - 'nearest' -> retorno {0,1}, sólo para compatibilidad/velocidad
    Returns float32 in [0,1], shape ≈ ceil(H/stride) x ceil(W/stride).
    """
    if stride <= 1:
        return mask.astype(np.float32)

    # fuerza binaria de entrada, pero como 0/1 float
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape
    nh, nw = math.ceil(h / stride), math.ceil(w / stride)

    if method == "avgpool":
        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        if pad_h or pad_w:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = mask.shape
        view = mask.reshape(h // stride, stride, w // stride, stride).astype(np.float32)
        prob = view.mean(axis=(1, 3))  # promedio de 0/1 -> prob
        return prob.astype(np.float32)

    if method == "nearest":
        out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return out.astype(np.float32)

    # por defecto: area
    out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)  # ya produce fracciones
    return np.clip(out.astype(np.float32), 0.0, 1.0)