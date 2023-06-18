from io import BytesIO
from base64 import b64decode
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from transformers import ViTFeatureExtractor, ViTModel
from sentence_transformers import SentenceTransformer


class MultiModalDataset(Dataset):
    def __init__(self, img_txt_file, label_img_txt_file, raw_txt_file, label_raw_txt_file, filter_file, img_transform=None):
        self.img_transform = img_transform
        self.product_images, self.product_texts = self.load_data(img_txt_file, raw_txt_file)
        self.label_images, self.label_texts = self.load_data(label_img_txt_file, label_raw_txt_file)
        self.label_product_mapping = self.load_mapping(filter_file)
        self.img_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.img_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.text_model = SentenceTransformer('bert-base-nli-mean-tokens')

    def load_data(self, img_txt_file, raw_txt_file):
        images = {}
        texts = {}

        with open(img_txt_file, 'r') as f:
            for line in f:
                id_, img_url = line.strip().split()
                if id_ not in images:
                    images[id_] = []
                images[id_].append(self.load_image(img_url))

        with open(raw_txt_file, 'r') as f:
            for line in f:
                id_, text = line.strip().split("->")[:2]
                if id_ in images: # for now only as their are few images; todo: remove this contraint later
                    texts[id_] = text

        return images, texts

    def load_image(self, b64_encoded_img):
        img_data = b64decode(b64_encoded_img)
        img_buffer = BytesIO(img_data)
        image = Image.open(img_buffer)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image

    def load_mapping(self, filter_file):
        mapping = {}

        with open(filter_file, 'r') as f:
            for line in f:
                product_id, label_id = line.strip().split()
                if label_id not in mapping:
                    mapping[label_id] = []
                mapping[label_id].append(product_id)

        return mapping

    def __len__(self):
        return len(self.label_texts)

    def __getitem__(self, idx):
        label_id = str(idx)
        product_ids = self.label_product_mapping[label_id]
        product_image_vectors = []
        product_text_vectors = []

        with torch.no_grad():
            # Process product data
            for product_id in product_ids:
                # Process product image vectors
                product_images = self.product_images.get(product_id, [])
                for image in product_images:
                    if self.img_transform is not None:
                        image = self.img_transform(image)
                    image_features = self.encode_image(image)
                    product_image_vectors.append(image_features)

                # Process product text vectors
                product_text = self.product_texts.get(product_id, "")
                text_features = self.encode_text(product_text)
                product_text_vectors.append(text_features)

            # Process label image vectors
            label_images = self.label_images.get(label_id, [])
            label_image_vectors = [self.encode_image(image) for image in label_images]

            # Process label text vector
            label_text = self.label_texts.get(label_id, "")
            label_text_vector = self.encode_text(label_text)

        return product_image_vectors, product_text_vectors, label_image_vectors, label_text_vector

    def encode_image(self, image):
        image = image.unsqueeze(0)  # Add batch dimension
        image_features = self.img_feature_extractor(image)['pixel_values']
        image_features = self.img_model(image_features).last_hidden_state.squeeze(0)
        return image_features

    def encode_text(self, text):
        text_features = self.text_model.encode([text])
        return text_features



if __name__ == "__main__":
    data_dir =  Path("data/MM-AmazonTitles-300K")

    transform = ToTensor()  # You can modify the transform based on your requirements

    dataset = MultiModalDataset(data_dir/"img_encoded/img.bin",
                                data_dir/"img_encoded/img.bin", 
                                data_dir/"raw_data/train.raw.txt",
                                data_dir/"raw_data/label.raw.txt",
                                data_dir/'filter_labels_train.txt', 
                                img_transform=transform)

# 
