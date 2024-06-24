import os
from datasets import load_dataset

# Define the path to save the dataset
data_dir = '../data'

# Create the data directory if it does not exist
os.makedirs(data_dir, exist_ok=True)

# Load the clarin-pl/poquad dataset
dataset = load_dataset('clarin-pl/poquad')

# Save the dataset to the specified directory
dataset.save_to_disk(data_dir)

print(f"Dataset downloaded and saved to {data_dir}")
