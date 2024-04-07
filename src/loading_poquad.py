from datasets import load_dataset

# Download and import clarin-pl/poquad from huggingface datasets
poquad = load_dataset("clarin-pl/poquad")

poquad["train"].to_json("../data/poquad_train.json", orient="records", lines=True)
poquad["validation"].to_json("../data/poquad_validation.json", orient="records", lines=True)