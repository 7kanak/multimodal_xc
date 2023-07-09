import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np


class SiameseDataset(Dataset):
    def __init__(self, csv_file: str, columns: tuple[str, str], max_length = 32, model_name="bert-base-uncased"):
        data = self.load_data_from_file(csv_file)
        self.documents = data[columns[0]]
        self.labels = data[columns[1]]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.document_col = columns[0]
        self.label_col = columns[1]


    def load_data_from_file(self, csv_file):
        data = pd.read_csv(csv_file)
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)
        return data


    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents[index]
        label = self.labels[index]

        # Tokenize the document and product
        # document_tokens = self.tokenizer.encode(document, truncation=True, padding='max_length', max_length=self.max_length)
        # label_tokens = self.tokenizer.encode(label, truncation=True, padding='max_length', max_length=self.max_length)
 
        # # Convert tokens to PyTorch tensors
        # document_tensors = torch.tensor(document_tokens)
        # label_tensors = torch.tensor(label_tokens)

        return {"document_text":document, "label_text":label}


if __name__ == "__main__":
    sd = SiameseDataset("old_files/adbase_indexed_06062023.csv",columns= ("ML_Transcripted_text", "Actual_product"))
    print(sd[0])