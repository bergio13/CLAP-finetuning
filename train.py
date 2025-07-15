"""
Main script to orchestrate the fine-tuning process.
"""

import torch
import config
from data_loader import get_data_loaders
from model import build_model
from trainer import Trainer

def main():
    print(f"Using device: {config.DEVICE}")

    # Get DataLoaders and class weights
    train_loader, val_loader, class_weights = get_data_loaders(config)

    # Build Model
    model, feature_extractor = build_model(config)

    # Define Optimizer, Scheduler, and Loss Function
    # Separate parameters for different learning rates
    head_params = model.classification_head.parameters()
    encoder_params = [p for name, p in model.named_parameters() if p.requires_grad and 'classification_head' not in name]

    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': config.LEARNING_RATE_HEAD},
        {'params': encoder_params, 'lr': config.LEARNING_RATE_ENCODER}
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config
    )
    
    trainer.train()
    
    # Print parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n--- Model Parameter Summary ---")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

if __name__ == '__main__':
    main()
