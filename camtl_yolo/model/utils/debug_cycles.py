# debug_cycles.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch.nn as nn

def find_module_cycles(root: nn.Module) -> List[List[str]]:
    """
    DFS over ._modules to find cycles. Returns list of qualified-name trails.
    """
    seen: Dict[int, int] = {}  # 0=unseen, 1=visiting, 2=done
    trail: List[str] = []
    cycles: List[List[str]] = []

    def dfs(mod: nn.Module, qname: str):
        mid = id(mod)
        state = seen.get(mid, 0)
        if state == 1:
            # back-edge: cut trail from first occurrence
            try:
                i = trail.index(qname)
            except ValueError:
                i = 0
            cycles.append(trail[i:] + [qname + ".<cycle>"])
            return
        if state == 2:
            return
        seen[mid] = 1
        for name, child in mod._modules.items():
            if child is None:
                continue
            cq = f"{qname}.{name}"
            trail.append(cq)
            dfs(child, cq)
            trail.pop()
        seen[mid] = 2

    dfs(root, "root")
    return cycles

def assert_no_module_cycles(root: nn.Module) -> None:
    cs = find_module_cycles(root)
    if cs:
        msg = " ; ".join(" → ".join(p) for p in cs)
        raise RuntimeError(f"Module graph contains a cycle along path: {msg}")
