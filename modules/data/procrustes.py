import numpy as np
from numpy.linalg import svd
from typing import Tuple


def procrustes_align(
    E1      : np.ndarray,
    E2      : np.ndarray,
    src_ids : np.ndarray,
    tgt_ids : np.ndarray,
    R1      : np.ndarray = None,
    R2      : np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find orthogonal W that maps KG1's DICE space toward KG2's using
    training entity pairs as anchors, then apply it to all of E1.

    In TransE: h + r ≈ t. Procrustes finds the best rigid rotation/reflection
    W such that E1[anchor] @ W ≈ E2[anchor]. The same W is applied to ALL
    entities so structural relationships are preserved.

    If R1 and R2 are provided, also reports how well relation embeddings
    align after applying W (diagnostic only — W is learned from entities).

    Args:
        E1      : KG1 entity embeddings [N1, dim]
        E2      : KG2 entity embeddings [N2, dim]
        src_ids : training pair indices into E1  [n_train]
        tgt_ids : training pair indices into E2  [n_train]
        R1      : KG1 relation embeddings [n_rel1, dim]  (optional)
        R2      : KG2 relation embeddings [n_rel2, dim]  (optional)

    Returns:
        E1_aligned : E1 @ W  [N1, dim]
        W          : orthogonal transformation  [dim, dim]
    """
    A = E1[src_ids].astype(np.float64)
    B = E2[tgt_ids].astype(np.float64)

    M = A.T @ B
    U, s, Vt = svd(M, full_matrices=True)
    W = (U @ Vt).astype(np.float32)

    before = float(np.mean(np.linalg.norm(
        E1[src_ids] - E2[tgt_ids], axis=1
    )))
    after = float(np.mean(np.linalg.norm(
        E1[src_ids] @ W - E2[tgt_ids], axis=1
    )))

    print(f"\n[procrustes] ── DICE Space Pre-Alignment ──────")
    print(f"[procrustes] Anchor pairs        : {len(src_ids)}")
    print(f"[procrustes] Embedding dim       : {E1.shape[1]}")
    print(f"[procrustes] Mean L2 before      : {before:.4f}")
    print(f"[procrustes] Mean L2 after       : {after:.4f}")
    print(f"[procrustes] Improvement         : {(before - after) / before * 100:.1f}%")

    if R1 is not None and R2 is not None:
        R1_aligned = R1.astype(np.float32) @ W

        R1_norm = R1_aligned / (np.linalg.norm(R1_aligned, axis=1, keepdims=True) + 1e-8)
        R2_norm = R2.astype(np.float32)
        R2_norm = R2_norm / (np.linalg.norm(R2_norm, axis=1, keepdims=True) + 1e-8)

        sim = R1_norm @ R2_norm.T 
        top1_sim = sim.max(axis=1)

        print(f"\n[procrustes] ── Relation Alignment Quality ────")
        print(f"[procrustes] KG1 relations       : {R1.shape[0]}")
        print(f"[procrustes] KG2 relations       : {R2.shape[0]}")
        print(f"[procrustes] Mean top-1 sim      : {top1_sim.mean():.4f}")
        print(f"[procrustes] Frac sim > 0.8      : {(top1_sim > 0.8).mean():.2%}")
        print(f"[procrustes] Frac sim > 0.5      : {(top1_sim > 0.5).mean():.2%}")

    print(f"[procrustes] ───────────────────────────────────\n")

    return E1 @ W, W
