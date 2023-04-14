import os
from argparse import ArgumentParser

from main_module import FaceIDModule
from data_module import FaceIDDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(args):
    seed_everything(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    logger = WandbLogger(
        name=args.description,
        project="LiVSec",
    )

    checkpoint = ModelCheckpoint(monitor="accuracy", mode="max", save_last=True)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
    )

    model = FaceIDModule(args)
    data = FaceIDDataModule(args)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="LiVSec-FaceAuth")
    parser.add_argument("--data_path", type=str, default="/data/faceid")
    parser.add_argument("--preprocess_data", type=int, default=0, choices=[0, 1])

    # MODULE specific args
    parser = FaceIDModule.add_model_specific_args(parser)

    # DATA specific args
    parser = FaceIDDataModule.add_data_specific_args(parser)

    # TRAINER args
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--no_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args)
