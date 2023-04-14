from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from cnn_architecture.res_net import ResNet


class FaceIDModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.model = ResNet()
        self.loss = nn.CosineEmbeddingLoss(margin=0.5)
        self.cosine = nn.CosineSimilarity()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        return parser

    def get_stats(self, embed_x0, embed_x1, y):
        cosine = self.cosine(embed_x0, embed_x1)
        same_class_cosine = cosine[y == 1]
        diff_class_cosine = cosine[y == -1]
        true_positive = (same_class_cosine > self.hparams.threshold).sum()
        true_negative = (diff_class_cosine < self.hparams.threshold).sum()
        accuracy = (true_positive + true_negative) / float(y.size(0))
        return same_class_cosine.mean(), diff_class_cosine.mean(), accuracy

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0 = self(x0)
        embed_x1 = self(x1)
        loss = self.loss(embed_x0, embed_x1, y)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0 = self(x0)
        embed_x1 = self(x1)
        loss = self.loss(embed_x0, embed_x1, y)
        same_class_cos, diff_class_cos, acc = self.get_stats(embed_x0, embed_x1, y)
        self.log("loss/val", loss)
        self.log("cosine/same_class", same_class_cos)
        self.log("cosine/diff_class", diff_class_cos)
        self.log("accuracy", acc)

    def configure_optimizers(self):
        optimizer = SGD(
            self.model.parameters(),
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
