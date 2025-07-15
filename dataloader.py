"""
Handles loading, preprocessing, and batching of the Speech Commands dataset.
"""

import torch
import librosa
import numpy as np
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def process_speech_commands(target_sr, fixed_length, max_samples_per_split=None):
    """
    Loads, resamples, and pads/truncates the Speech Commands dataset.
    """
    ds, info = tfds.load('speech_commands',
                         split=['train', 'validation', 'test'],
                         with_info=True,
                         as_supervised=True)

    train_ds, val_ds, test_ds = ds
    original_sr = info.features['audio'].sample_rate

    def resample_and_fix_length(dataset, split_name, max_samples=None):
        audios, labels = [], []
        print(f"Processing {split_name} split (resampling to {target_sr} Hz)...")
        for i, (audio, label) in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            if i % 2000 == 0 and i > 0:
                print(f"  ...processed {i} samples.")

            audio_np = audio.numpy().astype(np.float32)
            resampled = librosa.resample(audio_np, orig_sr=original_sr, target_sr=target_sr)

            if len(resampled) > fixed_length:
                resampled = resampled[:fixed_length]
            elif len(resampled) < fixed_length:
                resampled = np.pad(resampled, (0, fixed_length - len(resampled)), 'constant')

            audios.append(resampled)
            labels.append(label.numpy())

        print(f"Finished processing {split_name} split. Total samples: {len(audios)}")
        return np.array(audios), np.array(labels)

    train_audio, train_labels = resample_and_fix_length(train_ds, "train", max_samples_per_split)
    val_audio, val_labels = resample_and_fix_length(val_ds, "validation", max_samples_per_split)
    test_audio, test_labels = resample_and_fix_length(test_ds, "test", max_samples_per_split)

    return (train_audio, train_labels), (val_audio, val_labels), (test_audio, test_labels)

class SpeechCommandsDataset(Dataset):
    """PyTorch Dataset for Speech Commands."""
    def __init__(self, audio_data, labels):
        self.audio_data = audio_data
        self.labels = labels

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return {
            "audio": torch.tensor(self.audio_data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_data_loaders(config):
    """
    Creates and returns train, validation, and test DataLoaders.
    """
    (train_audio, train_labels), (val_audio, val_labels), _ = process_speech_commands(
        target_sr=config.TARGET_SR,
        fixed_length=config.FIXED_LENGTH_SAMPLES,
        max_samples_per_split=config.MAX_SAMPLES_PER_SPLIT
    )

    train_dataset = SpeechCommandsDataset(train_audio, train_labels)
    val_dataset = SpeechCommandsDataset(val_audio, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for handling imbalance
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for i in range(config.NUM_CLASSES):
        weight = total_samples / (config.NUM_CLASSES * label_counts.get(i, 1))
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
    print(f"Calculated class weights: {class_weights.cpu().numpy()}")

    return train_loader, val_loader, class_weights
