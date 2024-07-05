import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# tokenize the examples
def convert_to_features(tokenizer, example_batch):
    # Max lengths might required readjustment
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings



class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length_input, max_length_label):
        self.inputs = data["input_text"]
        self.labels = data["target_text"]
        self.tokenizer = tokenizer
        self.max_length_input = max_length_input
        self.max_length_label = max_length_label

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs.iloc[idx]
        label_text = self.labels.iloc[idx]
        
        input_encoding = self.tokenizer(
            input_text, 
            max_length=self.max_length_input, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        label_encoding = self.tokenizer(
            label_text, 
            max_length=self.max_length_label, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': label_encoding['input_ids'].squeeze()  # Labels typically don't have attention_mask
        }
        return item


# def create_data_loader(dataframe, tokenizer, max_len, batch_size):
#     ds = TextDataset(
#         dataframe=dataframe,
#         tokenizer=tokenizer,
#         max_len=max_len
#     )
#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         num_workers=4
#     )

# Example usage:
# df = pd.read_csv('path_to_your_csv.csv')
# dataloader = load_data(df)

# Iterate through the DataLoader
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     target_ids = batch['target_ids']
#     target_attention_mask = batch['target_attention_mask']
#     # Your training code here
