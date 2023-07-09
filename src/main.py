import torch
import torch.nn as nn
import torch.optim as optim
from model_base import SiameseNetwork, predict
from sampler_base import CustomSampler
from dataset_base import SiameseDataset
from loss import construct_loss
from torch.utils.data import Subset, random_split




def get_loaders(dataset, model, cluster_size, batch_size):
    assert cluster_size <= batch_size//2

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    sampler = CustomSampler(train_dataset, model, cluster_size, batch_size)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler) # todo: update this
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=sampler) # todo: change
    return train_dataloader, val_dataloader

def get_cluster_size(curr_cluster_size, epoch, batch_size):
    return curr_cluster_size

def train_siamese_model(input_dataset):
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 16
    cluster_size = 4
    refresh_interval = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(device)  # Initialize your Siamese model
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criteria = construct_loss()
    model.to(device)
    
    
    train_dataloader, val_dataloader = None, None

    history = {"train_loss": [], "val_loss": []}
    model.train()
    
    for epoch in range(num_epochs):
        if epoch%refresh_interval==0:
            print("Refreshing clusters....")
            cluster_size = get_cluster_size(cluster_size, epoch, batch_size)
            train_dataloader, val_dataloader = get_loaders(input_dataset, model, cluster_size=cluster_size, batch_size=batch_size)

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
    
        # model.eval()
        # with torch.no_grad():
        #     for batch in val_dataloader:
        #         sim, target = model(batch)
        #         loss = criteria(input=sim, target=target)
        #         val_loss += loss.item()

        #     val_loss /= len(val_dataloader)
        #     history["val_loss"].append(val_loss)
        # print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    print("Training completed.")
    return model

def main():
    sd = SiameseDataset("old_files/adbase_indexed_06062023.csv",columns= ("ML_Transcripted_text", "Actual_product"))
    model = train_siamese_model(sd)
    
    import pandas as pd
    test_data = pd.read_csv("old_files/adbase_indexed_06062023.csv").sample(100)
    test_data = test_data.dropna()
    test_data_list = test_data["ML_Transcripted_text"].tolist()
    true_val = test_data["Actual_product"].tolist()

    all_labels = pd.read_csv("old_files/adbase_indexed_06062023.csv")["Actual_product"].tolist()


    predict(model, test_data_list, all_labels, true_val)


if __name__ == "__main__":
    main()
