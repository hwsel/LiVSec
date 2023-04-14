import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class AttackFaceIDDataset(Dataset):
    def __init__(self, args, split, mode=None):
        self.hparams = args
        self.split = split
        self.mode = mode

        if self.split == "train":
            self.no_people = self.hparams.no_people_train
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomResizedCrop(self.hparams.img_size, scale=(0.5, 1.0)),
                ]
            )

        else:
            self.no_people = self.hparams.no_people_val
            self.transform = T.CenterCrop(self.hparams.img_size)

    def __len__(self):
        return self.hparams.poses_per_person * self.no_people

    def __getitem__(self, index):
        # First photo is deterministic. Second photo is random.
        # Get 1st photo
        real_index = index % (self.no_people * self.hparams.poses_per_person)
        person_id = real_index // self.hparams.poses_per_person
        pose_id = real_index % self.hparams.poses_per_person

        if self.mode == "result_collect":
            x_ref = torch.load(
                os.path.join(
                    self.hparams.data_path,
                    self.split,
                    "person" + str(person_id) + "_pose" + str(18) + ".pt",
                )
            )
            x = torch.load(
                os.path.join(
                    self.hparams.data_path,
                    self.split,
                    "person" + str(person_id) + "_pose" + str(pose_id) + ".pt",
                )
            )
        else:
            x_ref = torch.load(
                os.path.join(
                    self.hparams.data_path,
                    self.split,
                    "person" + str(person_id) + "_pose" + str(pose_id) + ".pt",
                )
            )

            # Get 2nd photo
            pose2_valid_choices = list(
                set(range(self.hparams.poses_per_person)) - set([pose_id])
            )
            pose2_id = pose2_valid_choices[
                torch.randint(len(pose2_valid_choices), (1,)).item()
            ]

            x = torch.load(
                os.path.join(
                    self.hparams.data_path,
                    self.split,
                    "person" + str(person_id) + "_pose" + str(pose2_id) + ".pt",
                )
            )

        x_ref = self.transform(x_ref)
        x = self.transform(x)
        x_mask = (x[3] < 0.9999).unsqueeze(0)  # after fixing bug in preprocessing, the background is 1
        return x_ref, x, x_mask


class AttackFaceIDDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_size", type=int, default=256)
        parser.add_argument("--no_people_train", type=int, default=26)
        parser.add_argument("--no_people_val", type=int, default=5)
        parser.add_argument("--poses_per_person", type=int, default=51)
        return parser

    def train_dataloader(self):
        dataset = AttackFaceIDDataset(self.hparams, "train")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.no_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = AttackFaceIDDataset(self.hparams, "val")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.no_workers,
            pin_memory=True,
        )
