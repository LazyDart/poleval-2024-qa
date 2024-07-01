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
    def __init__(self, dataframe, tokenizer, max_len_input, max_len_target):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input_text = dataframe['input_text']
        self.target_text = dataframe['target_text']
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = self.input_text.loc[index]
        target_text = self.target_text.loc[index]

        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len_input,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        
        targets = self.tokenizer.encode_plus(
            target_text,
            add_special_tokens=True,
            max_length=self.max_len_target,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids'],
            'decoder_attention_mask': targets['attention_mask']
        }
    

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
