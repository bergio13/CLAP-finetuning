"""
Configuration file for model fine-tuning.
Stores all hyperparameters and settings in one place.
"""

import torch

# --- Training Hyperparameters ---
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_ENCODER = 1e-5
WEIGHT_DECAY = 1e-5

# --- Model & Tokenizer ---
MODEL_NAME = "laion/clap-htsat-fused"

# --- Dataset Settings ---
TARGET_SR = 48000
FIXED_LENGTH_SAMPLES = 48000  # 1 second at 48kHz
MAX_SAMPLES_PER_SPLIT = 8000  # Use None for the full dataset
NUM_CLASSES = 12

# --- Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File Paths ---
MODEL_SAVE_PATH = 'clap_speech_commands_model.pth'
