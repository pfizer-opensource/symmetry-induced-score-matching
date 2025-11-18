import os
from argparse import ArgumentParser
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


if __name__ == "__main__":

    from sism.plcomplex.hparams import add_arguments
    from sism.plcomplex.data import LigandPocketDataModule as DataModule
    from sism.plcomplex.model import TrainerSphere
    from sism.plcomplex.model import TrainerRSGM
    from sism.plcomplex.model import TrainerFisherBridge

    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

    datamodule = DataModule(hparams)
    datamodule.setup("fit")

    if hparams.model == "gsm":
        print("Training generalized score matching")
        model = TrainerSphere(hparams=hparams.__dict__)
    elif hparams.model == "rsgm":
        print("Training Riemannian SGM")
        model = TrainerRSGM(hparams=hparams.__dict__)
    elif hparams.model == "fisher_bridge":
        print("Training Fisher Bridge")
        model = TrainerFisherBridge(hparams=hparams.__dict__)
    else:
        raise ValueError(
            f"Unknown model type: {hparams.model}. Choose from 'gsm', 'rsgm', or 'fisher_bridge'."
        )

    from lightning.pytorch.plugins.environments import LightningEnvironment

    strategy = "ddp" if hparams.gpus > 1 else "auto"
    strategy = "auto"
    callbacks = [
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus,  # if hparams.gpus else 1],
        strategy=strategy,
        plugins=LightningEnvironment(),
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=hparams.load_ckpt if hparams.load_ckpt != "" else None,
    )
