import torch
import numpy as np
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
        embeddings = F.normalize(embeddings , p=2, dim=-1)
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

        # x_hat_1 = F.normalize(x_hat_1 , p=2, dim=-1)
        # y_hat_1 = F.normalize(y_hat_1 , p=2, dim=-1)

        similarity = x_hat_1.squeeze() @ y_hat_1.squeeze().T

        target = torch.eye(similarity.shape[0]).to(self.device)
        return similarity, target


def predict(model: SiameseNetwork, input_document: list, all_labels: list, true_val): # todo: cache all_labels_rep
    model.eval()
    model.text_model.eval()
    model = model.cpu()
    model.text_model = model.text_model.cpu()
    model.device = "cpu"
    model.text_model.device = "cpu"
    with torch.no_grad():
        input_document = [i for i in input_document if isinstance(i, str)]
        input_doc_rep = model.text_model(input_document) # eg. if prediction for 100 doc, then of size 100,1
        all_labels = [i for i in all_labels if isinstance(i, str)]

        all_labels_rep = []
        chunks = [all_labels[i:i+100] for i in range(0, len(all_labels), 100)]
        for i in chunks:
            tmp_rep = model.text_model(i) # eg. if 1000 total labels then of size 1000,1
            all_labels_rep.append(tmp_rep.squeeze().numpy())
        all_labels_rep = np.vstack(all_labels_rep)
        all_labels_rep = torch.Tensor(all_labels_rep)

        sim = input_doc_rep.squeeze() @ all_labels_rep.T
        res = (torch.argsort(sim, dim=1)[:, -1]).numpy()
        correct = np.array([all_labels.index(i) for i in true_val])
        print(np.mean(res==correct))

    return res
