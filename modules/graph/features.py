import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import re
from sentence_transformers import SentenceTransformer


# === Basic helpers ===
#def _cosine_topk(mat_src, mat_tgt, src_uris, tgt_uris, k=10):
    #X = torch.tensor(mat_src, dtype=torch.float32)
    #Y = torch.tensor(mat_tgt, dtype=torch.float32)
    #X = F.normalize(X, dim=1)
    #Y = F.normalize(Y, dim=1)
    #sim = X @ Y.T
    #vals, idxs = torch.topk(sim, k=k, dim=1)
    #out = {src_uris[i]: [(tgt_uris[j.item()], vals[i, c].item()) for c, j in enumerate(idxs[i])]
           #for i in range(len(src_uris))}
    #return out



