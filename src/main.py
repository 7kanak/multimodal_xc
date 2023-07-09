import torch
import torch.nn as nn
import torch.optim as optim
from model_base import SiameseNetwork
from sampler_base import CustomSampler
from dataset_base import SiameseDataset
from loss import construct_loss




def get_loaders(dataset, model, cluster_size, batch_size):
    assert cluster_size <= batch_size//2
    sampler = CustomSampler(dataset, model, cluster_size, batch_size)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler) # todo: change
    return train_dataloader, val_dataloader

def get_cluster_size(curr_cluster_size, epoch, batch_size):
    return curr_cluster_size

def train_siamese_model():
    
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 16
    cluster_size = 4
    refresh_interval = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(device)  # Initialize your Siamese model
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criteria = construct_loss()
    model.to(device)
    
    sd = SiameseDataset("old_files/adbase_indexed_06062023.csv",columns= ("ML_Transcripted_text", "Actual_product"))
    train_dataloader, val_dataloader = None, None

    history = {"train_loss": [], "val_loss": []}
    model.train()
    
    for epoch in range(num_epochs):
        if epoch%refresh_interval==0:
            print("Refreshing clusters....")
            cluster_size = get_cluster_size(cluster_size, epoch, batch_size)
            train_dataloader, val_dataloader = get_loaders(sd, model, cluster_size=cluster_size, batch_size=batch_size)

        train_loss = 0.0
        val_loss = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            sim, target = model(batch)
            loss = criteria(input=sim, target=target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        history["train_loss"].append(train_loss)
    
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                loss = model(batch)
                val_loss += loss.item()

            val_loss /= len(val_dataloader)
            history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    print("Training completed.")

def main():
    train_siamese_model()


if __name__ == "__main__":
    main()
