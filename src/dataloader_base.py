import torch
from torch.utils.data import DataLoader
from model_base import SiameseNetwork
from dataset_base import SiameseDataset
from clustering import cluster_balance, b_kmeans_dense
import numpy as np
from torch.utils.data import Sampler


class ClusterSampler(Sampler):
    def __init__(self, cluster_indices, batch_size, shuffle=True):
        self.cluster_indices = cluster_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.cluster_indices))
        else:
            indices = torch.arange(len(self.cluster_indices))

        selected_clusters = self.cluster_indices[indices]
        samples = []

        for cluster in selected_clusters:
            samples.extend(cluster.tolist())

        num_batches = len(samples) // self.batch_size
        for i in range(num_batches):
            batch_samples = samples[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_samples

    def __len__(self):
        return len(self.cluster_indices)


def get_dataloader():
    # Assuming you have already created your train_dataset and model objects
    num_clusters = 100  # Specify the number of clusters
    batch_size = 32  # Specify the desired batch size

    sampler = ClusterSampler(cluster_indices, batch_size)

    # Create the CustomDataLoader
    dataloader = CustomDataLoader(train_dataset, model, num_clusters, batch_size, shuffle=True)

    # Pass the sampler to the DataLoader
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

    return dataloader


class CustomDataLoader:
    def __init__(self, train_dataset: SiameseDataset, model: SiameseNetwork, C = 16, batch_size = 32):
        self.train_dataset = train_dataset
        self.model = model
        self.shuffle = True
        self.cluster_centers = None
        self.num_threads = 1

        self.cluster_size = C
        self.num_clusters = len(train_dataset)//self.cluster_size
        self.batch_size = batch_size


        self._cluster_data()

    def _cluster_data(self):
        train_data_loader = DataLoader(self.train_dataset, batch_size=512, shuffle=False)
        model_output = []
        with torch.no_grad():
            self.model.eval()
            for batch in train_data_loader:
                inputs = batch[self.train_dataset.document_col[0]] 
                output = self.model.text_model(inputs)
                model_output.append(output)
        X = np.array(model_output)

        self.index, _ = cluster_balance(
            X=X.astype('float32'), 
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=self.num_clusters,
            splitter=b_kmeans_dense,
            num_threads=self.num_threads,
            verbose=True)

    def __iter__(self):
        # Step 3: Randomly select B clusters for each iteration
        if self.shuffle:
            indices = torch.randperm(len(self.index))
        else:
            indices = torch.arange(len(self.index))
        selected_clusters = self.index[indices[:self.B]]
        

        samples = []
        for cluster_id in selected_clusters:
            cluster_samples = self.index[cluster_id]
            samples.extend(cluster_samples.tolist())
        # Return the selected clusters as the iterator output
        return iter(selected_clusters)

    def __len__(self):
        return len(self.cluster_centers)


if __name__ == "__main__":
    sd = SiameseDataset("old_files/adbase_indexed_06062023.csv",columns= ("ML_Transcripted_text", "Actual_product"))
    network = SiameseNetwork()
    cd = CustomDataLoader(train_dataset=sd, model=network, num_clusters=3, batch_size=32)

    print(cd)