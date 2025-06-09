from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from datasets import load_dataset
#import os
#import numpy as np

# This object will hold the preprocessor across calls
_preprocessor = None

def init_preprocessor(class_list, duration=5):
    global _preprocessor
    _preprocessor = MelSpectrogramPreprocessors(duration=duration, class_list=class_list)

def process_audio_file(path, label_ids):
    """
    Wrap single-sample batch logic to match Rust-style inputs
    """
    batch = {
        "audio": [{"path": path}],
        "labels": [label_ids],
    }
    result = _preprocessor(batch)

    # Save mel spectrogram to disk
    # mel_array = np.array(result["audio_in"][0])  # Shape: [1, H, W]
    # base_name = os.path.splitext(os.path.basename(path))[0]
    # os.makedirs("mel_outputs", exist_ok=True)
    # np.save(f"mel_outputs/{base_name}_mel.npy", mel_array)


    return {
        "mel": result["audio_in"][0].tolist(),  # Shape: [1, H, W]
        "labels": result["labels"][0].tolist(), # Shape: [num_classes]
    }
