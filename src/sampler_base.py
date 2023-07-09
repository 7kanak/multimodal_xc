import torch
from torch.utils.data import Sampler
from model_base import SiameseNetwork
from dataset_base import SiameseDataset
from clustering import cluster_balance, b_kmeans_dense
import numpy as np
from torch.utils.data import DataLoader


class CustomSampler(Sampler):
    def __init__(self, train_dataset, model, cluster_size, batch_size):
        self.train_dataset = train_dataset
        self.model = model
        self.num_clusters = len(train_dataset)//cluster_size
        self.num_selected_clusters = batch_size//cluster_size
        self.cluster_indices = None
        self.num_batches = 2* len(train_dataset)//batch_size
        self._cluster_data()

    def _cluster_data(self):
        # Step 2: Pass all data through the model and cluster the outputs
        train_data_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False)
        model_output = []
        with torch.no_grad():
            self.model.eval()
            for batch in train_data_loader:
                inputs = batch["document_text"] 
                output = self.model.text_model(inputs)
                model_output.append(output.cpu().numpy()) # todo: optimize
        X = np.vstack(model_output)
        X = np.squeeze(X)
        self.cluster_indices, _  = cluster_balance(
                                            X=X.astype('float32'), 
                                            clusters=[np.arange(len(X), dtype='int')],
                                            num_clusters=self.num_clusters,
                                            splitter=b_kmeans_dense,
                                            num_threads=24,
                                            verbose=True)
        self.cluster_indices = [list(i) for i in self.cluster_indices]
            
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batches = []
        for i in range(self.num_batches):
            selected_cluster_indices = np.random.choice(
                                                range(len(self.cluster_indices)),
                                                size=self.num_selected_clusters,
                                                replace=False
                                                                )
            res = []
            for i in selected_cluster_indices:
                res.extend(self.cluster_indices[i])
            batches.append(res)
        return iter(batches)


if __name__ == "__main__":
    sd = SiameseDataset("old_files/adbase_indexed_06062023.csv",columns= ("ML_Transcripted_text", "Actual_product"))
    network = SiameseNetwork()

    sampler = CustomSampler(sd, network, 4, 16)
    x = list(sampler)
    dataloader = torch.utils.data.DataLoader(sd, batch_sampler=sampler)
    for batch in dataloader:
        print(batch)