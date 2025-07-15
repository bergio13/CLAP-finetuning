"""
Contains the Trainer class for handling the model training and validation loops.
"""

import torch
from tqdm import tqdm
import copy

class Trainer:
    def __init__(self, model, feature_extractor, train_loader, val_loader, optimizer, scheduler, criterion, config):
        self.model = model
        self.feature_extractor = feature_extractor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = config.DEVICE
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _process_batch(self, batch):
        audios = batch["audio"]
        labels = batch["label"].to(self.device)
        
        inputs = self.feature_extractor(
            audios,
            sampling_rate=self.config.TARGET_SR,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs, labels

    def _train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch in tqdm(self.train_loader, desc="Training"):
            inputs, labels = self._process_batch(batch)
            
            self.optimizer.zero_grad()
            
            audio_embeddings = self.model.get_audio_features(**inputs)
            logits = self.model.classification_head(audio_embeddings)
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * total_correct / total_samples
        return avg_loss, accuracy

    def _validate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = self._process_batch(batch)

                audio_embeddings = self.model.get_audio_features(**inputs)
                logits = self.model.classification_head(audio_embeddings)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * total_correct / total_samples
        return avg_loss, accuracy

    def train(self):
        print("Starting training...")
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"✨ New best model found! Saving weights. (Val Loss: {self.best_val_loss:.4f})")

        print("\nTraining complete.")
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
            print(f"Best model saved to {self.config.MODEL_SAVE_PATH}")
