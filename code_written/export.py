import os
import json
import pandas as pd
from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
#from extractors.birdset import Birdset

# Load BirdSet for region HSN
birdset_extactor = extractors.Birdset()
dataset = birdset_extactor("HSN")  # returns AudioDataset with .splits dict

# Directory to save outputs
output_dir = "birdset_export/HSN"
os.makedirs(output_dir, exist_ok=True)

def save_split_to_json(split_name, data):
    # Convert HF Dataset to Pandas, then to dict
    df = data.to_pandas()
    # Save as JSON
    json_path = os.path.join(output_dir, f"{split_name}.json")
    df.to_json(json_path, orient="records", lines=False, indent=2)
    print(f"Saved {split_name} to {json_path}")

def save_split_to_csv(split_name, data):
    df = data.to_pandas()
    csv_path = os.path.join(output_dir, f"{split_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {split_name} to {csv_path}")

# Export each split
for split_name in dataset.data.keys():
    split_data = dataset.data[split_name]
    save_split_to_json(split_name, split_data)
    save_split_to_csv(split_name, split_data)

