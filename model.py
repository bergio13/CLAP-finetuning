"""
Handles the creation and modification of the CLAP model for classification.
"""

import torch
from transformers import ClapModel, ClapFeatureExtractor

def build_model(config):
    """
    Builds the CLAP model, adds a classification head, and freezes layers.
    """
    print(f"Loading pretrained model: {config.MODEL_NAME}")
    model = ClapModel.from_pretrained(config.MODEL_NAME)
    feature_extractor = ClapFeatureExtractor.from_pretrained(config.MODEL_NAME)
    
    # Add a new classification head
    audio_embedding_dim = model.audio_projection.linear2.out_features
    model.classification_head = torch.nn.Linear(audio_embedding_dim, config.NUM_CLASSES)

    # Initialize the new head's weights
    torch.nn.init.xavier_uniform_(model.classification_head.weight)
    torch.nn.init.zeros_(model.classification_head.bias)
    
    print("Added new classification head.")

    # Freeze text-related parameters
    for name, param in model.named_parameters():
        if name.startswith('text_model') or name.startswith('text_projection'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Ensure the new head is trainable
    for param in model.classification_head.parameters():
        param.requires_grad = True
        
    print("Froze text encoder. Audio encoder and classification head are trainable.")
    
    model.to(config.DEVICE)
    return model, feature_extractor
