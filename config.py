import torch
from collections import Counter

NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SAMPLE_RATE = 48000
FIXED_LENGTH = 48000
NUM_CLASSES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
