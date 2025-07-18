import torch
from transformers import ClapModel, ClapFeatureExtractor
from config import DEVICE, NUM_CLASSES

def initialize_model():
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    audio_dim = model.audio_projection.linear2.out_features
    model.classification_head = torch.nn.Linear(audio_dim, NUM_CLASSES)
    torch.nn.init.xavier_uniform_(model.classification_head.weight)
    torch.nn.init.zeros_(model.classification_head.bias)

    for name, param in model.named_parameters():
        if name.startswith("text_model") or name == "logit_scale_t":
            param.requires_grad = False

    model.to(DEVICE)
    return model

def get_feature_extractor():
    return ClapFeatureExtractor.from_pretrained("laion/clap-htsat-fused")
