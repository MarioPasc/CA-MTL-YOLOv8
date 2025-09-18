# camtl_yolo/train/samplers.py
"""
Batch schedulers and samplers.

- AlternatingLoader: yield batches from two loaders in ratio X:Y.
- domain helpers: map free-form domain strings to {"source","target"}.
"""
from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Tuple
import itertools


def map_domain_name(name: str) -> str:
    n = (name or "").lower()
    if "retino" in n or "fundus" in n or "source" in n:
        return "source"
    return "target"


class AlternatingLoader:
    """
    Iterate over two torch DataLoaders with ratio X:Y (det:seg).

    Example
    -------
    alt = AlternatingLoader(det_loader, seg_loader, ratio=(1,2))
    for b in alt:
        ...
    """
    def __init__(self, det_loader, seg_loader, ratio: Tuple[int, int] = (1, 1), length: int | None = None):
        self.det_loader = det_loader
        self.seg_loader = seg_loader
        self.rx, self.ry = int(ratio[0]), int(ratio[1])
        # define nominal length (number of yielded batches per epoch)
        if length is None:
            # min number of full patterns supported by both loaders
            len_det = len(det_loader)
            len_seg = len(seg_loader)
            # number of patterns limited by smaller effective stream
            patterns = min(len_det // max(self.rx, 1), len_seg // max(self.ry, 1))
            length = max(1, patterns * (self.rx + self.ry))
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Dict]:
        det_iter = itertools.cycle(iter(self.det_loader))
        seg_iter = itertools.cycle(iter(self.seg_loader))
        count = 0
        while count < self._length:
            for _ in range(self.rx):
                batch = next(det_iter)
                batch["bn_domain"] = [map_domain_name(d) for d in batch.get("domain", [])]
                yield batch
                count += 1
                if count >= self._length:
                    return
            for _ in range(self.ry):
                batch = next(seg_iter)
                batch["bn_domain"] = [map_domain_name(d) for d in batch.get("domain", [])]
                yield batch
                count += 1
                if count >= self._length:
                    return
