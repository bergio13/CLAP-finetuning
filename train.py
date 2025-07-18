import torch
from config import *
from validate import validate_model
from utils import get_class_weights, clip_audio_batch
from data import batch_data
from collections import Counter

def train(model, train_data, val_data, feature_extractor):
    print(f"Using device: {DEVICE}")
    criterion = torch.nn.CrossEntropyLoss(weight=get_class_weights(train_data[1]))

    classification_params = [p for n, p in model.named_parameters() if "classification_head" in n and p.requires_grad]
    encoder_params = [p for n, p in model.named_parameters() if "classification_head" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": classification_params, "lr": LEARNING_RATE},
        {"params": encoder_params, "lr": LEARNING_RATE * 0.1}
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        model.train()
        total_loss, correct, total = 0, 0, 0
        batch_count = 0
        all_preds, all_labels = [], []

        for audio_batch, label_batch in batch_data(list(zip(train_data[0], train_data[1])), batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            audio_batch = clip_audio_batch(audio_batch)
            inputs = feature_extractor(audio_batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = torch.tensor(label_batch, dtype=torch.long).to(DEVICE)

            embeddings = model.get_audio_features(**inputs)
            logits = model.classification_head(embeddings)

            loss = criterion(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_correct = (logits.argmax(dim=1) == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            batch_count += 1

            preds_np = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds_np)
            all_labels.extend(labels.cpu().numpy())

            if batch_count % 10 == 0:
                acc = 100.0 * correct / total
                print(f"[Epoch {epoch+1} | Batch {batch_count}] Loss: {loss.item():.4f} | Acc: {acc:.2f}% | Grad Norm: {grad_norm:.4f}")
                print(f"Sample predictions: {preds_np[:5]} | True: {labels.cpu().numpy()[:5]}")

        scheduler.step()
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        acc = 100.0 * correct / total if total > 0 else 0.0
        pred_dist = Counter(all_preds)
        label_dist = Counter(all_labels)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
        print(f"Top predictions: {pred_dist.most_common(3)}")
        print(f"Top labels: {label_dist.most_common(3)}")

        print("Running validation...")
        val_loss, val_acc = validate_model(model, val_data, feature_extractor, criterion)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print("New best model saved!")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights.")

    return model
