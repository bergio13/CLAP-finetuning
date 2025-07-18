import torch
from config import DEVICE, BATCH_SIZE, SAMPLE_RATE
from utils import clip_audio_batch, batch_data

def validate_model(model, val_data, feature_extractor, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for audio_batch, label_batch in batch_data(list(zip(val_data[0], val_data[1])), batch_size=BATCH_SIZE):
            audio_batch = clip_audio_batch(audio_batch)
            inputs = feature_extractor(audio_batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = torch.tensor(label_batch, dtype=torch.long).to(DEVICE)

            embeddings = model.get_audio_features(**inputs)
            logits = model.classification_head(embeddings)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total // BATCH_SIZE)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
