from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from cnn_architecture.u_net import UNet
from modules.faceid_module import FaceIDModule


class AttackFaceIDModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.attack_model = UNet(
            self.hparams.pert_strength,
        )
        self.faceid_model = FaceIDModule.load_from_checkpoint(
            "modules/livsec_face_auth.ckpt"
        )
        self.faceid_model.freeze()

        self.cosine = nn.CosineSimilarity()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--pert_strength", type=float, default=16.0 / 255.0)
        return parser

    def forward(self, x, x_mask):
        x_adv = self.attack_model(x, x_mask)
        return x_adv

    def share_step(self, batch):
        x_ref, x, x_mask = batch
        x_adv = self.attack_model(x, x_mask)

        embed_x_adv = self.faceid_model(x_adv)
        with torch.no_grad():
            embed_x_ref = self.faceid_model(x_ref)

        score = self.cosine(embed_x_adv, embed_x_ref)

        loss = score.mean()
        attack_accuracy = (score < self.hparams.threshold).float().mean()
        return loss, attack_accuracy

    def training_step(self, batch, batch_nb):
        loss, attack_accuracy = self.share_step(batch)
        self.log("loss/train", loss, prog_bar=True)
        self.log("attack_accuracy/train", attack_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, attack_accuracy = self.share_step(batch)
        self.log("loss/val", loss, prog_bar=True)
        self.log("attack_accuracy/val", attack_accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = SGD(
            self.attack_model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        reduce_lr = [
            int(0.50 * self.hparams.max_epochs),
            int(0.75 * self.hparams.max_epochs),
            int(0.90 * self.hparams.max_epochs),
        ]
        scheduler = {
            "scheduler": MultiStepLR(
                optimizer,
                milestones=reduce_lr,
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
