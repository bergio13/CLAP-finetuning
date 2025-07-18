import torch
import numpy as np
from config import DEVICE, NUM_CLASSES
from collections import Counter

def get_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def clip_audio_batch(batch, max_length=480_000):
    clipped = []
    for audio in batch:
        if isinstance(audio, np.ndarray):
            audio = audio[:max_length]
        elif isinstance(audio, torch.Tensor):
            audio = audio[..., :max_length]
        clipped.append(audio)
    return clipped
