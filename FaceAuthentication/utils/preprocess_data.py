import os
import re
import shutil
import zipfile

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm as pbar

train_url = "https://rutgers.box.com/shared/static/vfaxy3y1otrvn5dbvbxdg3unmvyun7yg.zip"
val_url = "https://rutgers.box.com/shared/static/smgbvw9v5qemc196072z51x7kwfaxn4b.zip"


def download(url):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte
    t = pbar(total=total_size, unit="MiB", unit_scale=True)

    header = r.headers["content-disposition"]
    file_name = re.findall("filename\*=UTF-8''(.+)", header)[0]

    with open(file_name, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise Exception("Error, something went wrong")

    print("Download successful. Unzipping " + file_name)
    file_path = os.path.join(os.getcwd(), file_name)

    extract_dir = os.getcwd()
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
        print("Unzip file successful!")


def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def stack_data(raw_dir, processed_dir):
    """
    Stack RGB and D images to a torch tensor of shape 4 x 300 x 300
    Save images in format of personID_poseID
    """
    all_people = [x for x in os.listdir(raw_dir) if "2012" in x]
    for person_id, a_person in enumerate(pbar(all_people)):

        all_poses = [
            x for x in os.listdir(os.path.join(raw_dir, a_person)) if ".bmp" in x
        ]
        for pose_id, a_pose in enumerate(all_poses):
            photo_id = a_pose[:-5]  # drop 'c.bmp' in file name

            photo_rgb = photo_id + "c.bmp"
            photo_rgb = Image.open(os.path.join(raw_dir, a_person, photo_rgb))
            photo_rgb = np.array(photo_rgb.convert("RGB").resize((640, 480)))

            photo_depth = photo_id + "d.dat"
            photo_depth = np.loadtxt(os.path.join(raw_dir, a_person, photo_depth))

            # Valid range is from 400 to 3000
            # But most of the fact depth is from 400 to 1200
            photo_depth[photo_depth == -1] = 1200  # set invalid data to 1200
            photo_depth = np.clip(photo_depth, a_min=400, a_max=1200)

            # Scale to [0-1] for photo_depth
            photo_depth = (photo_depth - 400) / (1200 - 400)

            # Scale to [0-255]
            photo_depth = photo_depth * 255

            # Align RGB and D
            photo_depth = np.pad(photo_depth, [(0, 0), (20, 0)], mode="constant")
            photo_depth = photo_depth[:, :-20]
            photo_depth = np.expand_dims(photo_depth, -1)

            # Stack RGB and D
            rgbd = np.concatenate((photo_rgb, photo_depth), axis=2)
            rgbd = crop_center(rgbd, 300, 300)
            rgbd = np.float32(rgbd)
            rgbd = rgbd / 255.0  # normalize to range [0, 1]

            rgbd = torch.from_numpy(rgbd)  # 300 x 300 x 4
            rgbd = rgbd.permute(2, 0, 1)  # 4 x 300 x 300
            rgbd = (rgbd - 0.5) / 0.5  # normalize to range [-1, 1]

            # Save to disk
            name = "person" + str(person_id) + "_pose" + str(pose_id) + ".pt"
            torch.save(rgbd, os.path.join(processed_dir, name))


def preprocess_data(data_path):
    download(train_url)
    download(val_url)

    shutil.move("train_raw.zip", data_path)
    shutil.move("val_raw.zip", data_path)
    shutil.move("faceid_train_raw", data_path)
    shutil.move("faceid_val_raw", data_path)

    os.mkdir(os.path.join(data_path, "train"))
    stack_data(
        os.path.join(data_path, "faceid_train_raw"), os.path.join(data_path, "train")
    )

    os.mkdir(os.path.join(data_path, "val"))
    stack_data(
        os.path.join(data_path, "faceid_val_raw"), os.path.join(data_path, "val")
    )


if __name__ == '__main__':
    data_path = '/data/faceid'
    print("Preprocessing data...")
    preprocess_data(data_path)
    print("Finished preprocessing data.")