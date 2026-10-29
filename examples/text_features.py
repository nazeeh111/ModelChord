"""Versioned deterministic character representation, shared with the text demo.

No vocabulary fit, pretrained tokenizer, network access, or Python randomized hash.
"""
from __future__ import annotations
import hashlib
import numpy as np

def text_features(texts: list[str], dimension: int = 256) -> np.ndarray:
    features = np.zeros((len(texts),dimension),dtype=np.float32)
    for row,text in enumerate(texts):
        if not isinstance(text,str) or len(text)>10000:
            raise ValueError("Expected SMS strings of at most 10000 characters")
        data = text.encode("utf-8")
        grams = [bytes([b]) for b in data] + [data[i:i+2] for i in range(max(0,len(data)-1))]
        for gram in grams:
            h = hashlib.blake2b(gram,digest_size=8,person=b"MChordV1").digest()
            index = int.from_bytes(h[:4],"little") % dimension
            features[row,index] += 1. if h[4]&1 else -1.
        norm=np.linalg.norm(features[row])
        if norm>0:features[row]/=norm
    return features

def transform_text(texts: list[str],task: str) -> list[str]:
    if task=="clean":return texts
    if task=="uppercase":return [s.upper() for s in texts]
    if task=="leet":return [s.translate(str.maketrans("aeiosAEIOS","4310543105")) for s in texts]
    if task=="underscores":return [s.replace(" ","_") for s in texts]
    raise ValueError(task)
