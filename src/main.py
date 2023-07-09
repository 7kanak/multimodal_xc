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

def get_cluster_size(curr_cluster_size, epoch):
    if epoch%10==0 and epoch!=0:
        curr_cluster_size/=2
    curr_cluster_size = min(curr_cluster_size, 4)
    return curr_cluster_size

def train_siamese_model(input_dataset):
    num_epochs = 6
    learning_rate = 0.001
    batch_size = 256
    cluster_size = 64
    refresh_interval = 5
    data_file = "old_files/adbase_indexed_06062023.csv"
    test_file = "old_files/adbase_indexed_06062023.csv"

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
            cluster_size = get_cluster_size(cluster_size, epoch)
            train_dataloader, _ = get_loaders(input_dataset, model, cluster_size=cluster_size, batch_size=batch_size)
            test(model, data_file, test_file, ("ML_Transcripted_text", "Actual_product"))
            model.train()
            model.text_model.train()

        
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
    train_siamese_model(sd)
    


def test(model, data_csv_file: str, test_csv_file: str, columns: tuple[str, str]):
    import pandas as pd
    test_data = pd.read_csv(test_csv_file)
    test_data = test_data.dropna()
    test_data_list = test_data[columns[0]].tolist()
    true_val = test_data[columns[1]].tolist()

    all_labels = pd.read_csv(data_csv_file)[columns[1]].tolist()

    predict(model, test_data_list, all_labels, true_val)


if __name__ == "__main__":
    main()
