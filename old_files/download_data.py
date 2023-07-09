import os
import multiprocessing as mp
from PIL import Image
import numpy as np
import requests
import time
import tqdm
import re
import base64
from io import BytesIO
from PIL import Image, ImageOps
from pathlib import Path

img_size = (
    128,
    128,
)  # Stores images as 128x128 base64 encoding. Please modify if you want a higher resolution image.


def get_urls(in_file_name, train_file_name):
    images_urls = []
    file_names = []
    with open(in_file_name, "r", encoding="latin1") as f:
        for line in tqdm.tqdm(f):
            uid, url = line.strip().split("\t", 1)
            images_urls.append(url)
            file_names.append(uid)
    print("{} Images detected".format(len(images_urls)))

    zipped_img_urls = list(zip(images_urls, file_names))


    train_file_names = []
    with open(train_file_name, "r", encoding="latin1") as f:
        for line in tqdm.tqdm(f):
            uid = line.strip().split("->", 1)[0]
            train_file_names.append(uid)
            if len(train_file_names) > 1000:
                break

    zipped_img_urls = [i for i in zipped_img_urls if i[1] in train_file_names]

    images_urls = [i[0] for i in zipped_img_urls]
    file_names = [i[1] for i in zipped_img_urls]

    print(f"length of files to download is {len(images_urls)}")
    return images_urls, file_names


def load_img(img, size=img_size):
    img = img.convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    final_size = img.size
    delta_w = size[0] - final_size[0]
    delta_h = size[1] - final_size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    l_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    with BytesIO() as im_file:
        l_img.save(im_file, format="JPEG")
        value = base64.b64encode(im_file.getvalue()).decode("utf-8")
        im_file.close()
    return value


def image_downloader(img_url):
    
    key, img_url = img_url
    res = requests.get(img_url, stream=True)
    count = 1
    while res.status_code != 200 and count <= 5:
      
        res = requests.get(img_url, stream=True)
        print(f"Retry: {count} {img_url}")
        count += 1
    # checking the type for image
    if "image" not in res.headers.get("content-type", ""):
        print("ERROR: URL doesnot appear to be an image")
        return None, False
    return key, load_img(Image.open(BytesIO(res.content)))


def write_to_base64(image_keys, images_url, num_process=4, fpath="images"):
    os.makedirs(fpath, exist_ok=True)
    f = open(f"{fpath}/img.bin", "wb")
    with mp.Pool(num_process) as p:
        for key, img in tqdm.tqdm(
            p.imap(image_downloader, zip(image_keys, images_url)), total=len(image_keys)
        ):
            if key is not None:
                f.write(bytes(f"{key}\t{img}\n".encode("utf-8")))
    f.close()


if __name__ == "__main__":
    data_dir =  Path("data/MM-AmazonTitles-300K")
    for inst in ["train", "test"]:
        images_url, image_keys = get_urls(data_dir / "img.urls.txt", data_dir / f"raw_data/{inst}.raw.txt")
        write_to_base64(image_keys, images_url,num_process=2, fpath=data_dir / f"img_encoded/{inst}")

    # download only matching labels

    with open(data_dir/'filter_labels_train.txt', "r") as f:
        