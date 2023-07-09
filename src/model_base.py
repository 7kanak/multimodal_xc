import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, device):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        self.model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-cos-v5")
        self.adaptive_pool = torch.nn.AdaptiveMaxPool1d(192)
        self.device = device
        

    def forward(self, text):
        encoded_inputs = self.tokenizer(text, 
                              max_length=64, truncation=True, 
                              padding="max_length", return_tensors="pt")
        input_ids = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings from the [CLS] token
        embeddings = self.adaptive_pool(embeddings)
        embeddings = embeddings.unsqueeze(1)
        return embeddings
    


class SiameseNetwork(torch.nn.Module):
    def __init__(self, device):
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.text_model = TextEncoder(device=device)

    def forward(self, x):
        xt = x["document_text"]
        yt = x["label_text"]
        
        x_hat_1 = self.text_model(xt)
        y_hat_1 = self.text_model(yt)

        x_hat_1 = F.normalize(x_hat_1 , p=2, dim=-1)
        y_hat_1 = F.normalize(y_hat_1 , p=2, dim=-1)

        similarity = x_hat_1.squeeze() @ y_hat_1.squeeze().T

        target = torch.eye(similarity.shape[0]).to(self.device)
        return similarity, target


