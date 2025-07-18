# CLAP-finetuning
To start the fine tuning
```python
from data import process_speech_commands
from model import initialize_model, get_feature_extractor
from train import train
from config import *

if __name__ == "__main__":
    print("Preparing data...")
    train_data, val_data, test_data = process_speech_commands(fixed_length=FIXED_LENGTH)

    print("Initializing model...")
    model = initialize_model()
    feature_extractor = get_feature_extractor()

    print("Starting training...")
    trained_model = train(model, train_data, val_data, feature_extractor)

    print("Saving model...")
    torch.save(trained_model.state_dict(), "clap_speech_commands_model.pth")
```
