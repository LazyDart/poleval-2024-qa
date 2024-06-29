import pandas as pd
from datasets import load_dataset, Dataset

import os
import json 


def download_poquad(data_dir):

    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Load the clarin-pl/poquad dataset
    dataset = load_dataset('clarin-pl/poquad')

    # Save the dataset to the specified directory
    dataset.save_to_disk(data_dir)

    print(f"Dataset downloaded and saved to {data_dir}")
  

def load_poquad_datasets(data_dir):
    # Define paths to train and validation data
    train_data_path = f"{data_dir}/train"
    validation_data_path = f"{data_dir}/validation"
    
    # Load datasets using Hugging Face datasets library
    train_dataset = Dataset.load_from_disk(train_data_path)
    validation_dataset = Dataset.load_from_disk(validation_data_path)
    
    return train_dataset, validation_dataset

def read_poquad_manually_downloaded(filepath):
        whole_data = []
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            id_ = 0
            for example in squad["data"]:
                title = example.get("title", "")
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        
                        if "answers" not in qa:
                            continue
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]

                        answers = [answer["text"] for answer in qa["answers"]]
                        is_impossible = qa["is_impossible"]

                        id_ += 1
                        whole_data.append({
                            "id": str(id_),
                            "title": title,
                            "context": context,
                            "question": question,
                            "is_impossible" : is_impossible,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        })

        return pd.DataFrame(whole_data).set_index("id")


def save_poquad_manually_downloaded(manual_train_path, manual_valid_path):
    if not os.path.exists("../data/poquad-manually-processed"):
        os.makedirs("../data/poquad-manually-processed")

    read_poquad_manually_downloaded(manual_valid_path).to_json("../data/poquad-manually-processed/poquad-valid.json")
    read_poquad_manually_downloaded(manual_train_path).to_json("../data/poquad-manually-processed/poquad-train.json")


def load_poquad_manually_downloaded(dirpath):
    train = pd.read_json(f"{dirpath}/poquad-train.json")
    valid = pd.read_json(f"{dirpath}/poquad-valid.json")

    return train, valid

def dataset_into_str_input(df):

    return (
        df["title"].apply(lambda x: "tytuł: " + x + " ")
        + df["context"].apply(lambda x: "kontekst: " + x + " ")
        + df["question"].apply(lambda x: "pytanie: " + x)
         )


# download_poquad(data_dir = '../data/poquad-original/')    

# load_poquad_datasets("../data/poquad-original/")