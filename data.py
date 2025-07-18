import tensorflow_datasets as tfds
import librosa
import numpy as np
import random

def process_speech_commands(target_sr=48000, max_samples_per_split=None, fixed_length=None):
    ds, info = tfds.load('speech_commands', split=['train', 'validation', 'test'], with_info=True, as_supervised=True)
    train_ds, val_ds, test_ds = ds
    original_sr = info.features['audio'].sample_rate

    if fixed_length is None:
        fixed_length = int(16000 * target_sr / original_sr)

    def resample_dataset(dataset, split_name, max_samples=None):
        audios, labels = [], []
        for i, (audio, label) in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            audio_np = audio.numpy().astype(np.float32)
            resampled = librosa.resample(audio_np, orig_sr=original_sr, target_sr=target_sr)
            resampled = resampled[:fixed_length] if len(resampled) > fixed_length else np.pad(resampled, (0, fixed_length - len(resampled)))
            audios.append(resampled)
            labels.append(label.numpy())
        return np.array(audios), np.array(labels)

    return (
        resample_dataset(train_ds, "train", max_samples_per_split),
        resample_dataset(val_ds, "validation", max_samples_per_split),
        resample_dataset(test_ds, "test", max_samples_per_split)
    )

def batch_data(data, batch_size, shuffle=False, drop_last=True):
    if shuffle:
        data = data.copy()
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if drop_last and len(batch) < batch_size:
            continue
        yield list(zip(*batch))
