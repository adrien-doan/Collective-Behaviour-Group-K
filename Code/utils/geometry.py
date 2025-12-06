import numpy as np

def normalize(vecs, eps=1e-8):
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / (norms + eps)