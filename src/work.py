# %%
import gc
import json
import os
import pickle
import random
import re
import string
import shutil
import sys
import time
from typing import Callable
import warnings
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import transformers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.simplefilter("ignore")

# ====================================================
# transform
# ====================================================


class Transform():
    def __init__(self, data):
        self.data = data
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        config = transformers.AutoConfig.from_pretrained(
            CFG.model.name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            CFG.model.name,
            config=config,
        )
        return tokenizer

    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=CFG.tokenizer.max_length,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

    def __call__(self, text):
        return self.tokenize(text)


def get_transform(data):
    return Transform(data=data)


# ====================================================
# optimizer
# ====================================================


def get_optimizer(model):
    return getattr(
        torch.optim,
        CFG.optimizer.name
    )(model.parameters(), **CFG.optimizer.params)

# ====================================================
# scheduler
# ====================================================


def get_scheduler(optimizer):
    return getattr(
        torch.optim.lr_scheduler,
        CFG.scheduler.name
    )(optimizer, **CFG.scheduler.params)


# ====================================================
# dataset
# ====================================================
class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        texts,
        transform
    ):
        self.texts = texts
        self.transform = transform
        self.length = len(texts)

    def __len__(
        self,
    ):
        return self.length

    def __getitem__(
        self,
        idx
    ):
        text = self.texts[idx]
        if self.transform:
            text = self.transform(text)
            text = {k: torch.tensor(v, dtype=torch.long)
                    for k, v in text.items()}
        return text


class Dataset(DatasetBase):
    def __init__(
        self,
        texts,
        labels,
        transform,
    ):
        super().__init__(texts, transform)
        self.labels = labels

    def __getitem__(
        self,
        idx,
    ):
        text = super().__getitem__(idx)
        label = torch.tensor(self.labels[idx]).float()
        return text, label


class TestDataset(DatasetBase):
    def __init__(
        self,
        texts,
        transform,
    ):
        super().__init__(texts, transform)

    def __getitem__(
        self,
        idx,
    ):
        text = super().__getitem__(idx)
        return text


# ====================================================
# model
# ====================================================
class Model(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            CFG.model.name,
            num_labels=1,
        )
        self.criteria = lambda y_hat, y: torch.sqrt(F.mse_loss(y_hat, y))
        self.history = {
            "train_loss": [],
            "valid_loss": [],
        }
        self.lr_history = []

    def forward(self, x):
        output = self.model(**x)
        return output.logits.flatten()

    def configure_optimizers(self):
        optimizer = get_optimizer(self)
        scheduler = get_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": CFG.scheduler.interval,
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criteria(y_hat, y)

        self.lr_history.append(
            self.lr_schedulers().get_lr()[0]
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criteria(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_epoch = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()
        self.log(
            name="train_loss_epoch",
            value=train_loss_epoch,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        self.history["train_loss"].append(
            train_loss_epoch.detach().cpu().numpy()
        )

    def validation_epoch_end(self, outputs):
        valid_loss_epoch = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()
        self.log(
            name="valid_loss_epoch",
            value=valid_loss_epoch,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        self.history["valid_loss"].append(
            valid_loss_epoch.detach().cpu().numpy()
        )

# ====================================================
# dataloader
# ====================================================


def get_dataloader(
    texts,
    labels,
    transform,
):
    ds = Dataset(texts, labels, transform)
    dl = DataLoader(ds, **CFG.loader.train)

    return dl


def get_test_dataloader(
    texts,
    transform,
):
    ds = TestDataset(texts, transform)
    dl = DataLoader(ds, **CFG.loader.test)

    return dl

# ====================================================
# plots
# ====================================================


def plot_dist(ytrue, ypred, filename):
    plt.figure()
    plt.hist(ytrue, alpha=0.5, bins=100)
    plt.hist(ypred, alpha=0.5, bins=100)
    plt.legend(["ytrue", "ypred"])
    plt.savefig(filename)
    plt.close()


def plot_training_curve(history, filename):
    plt.figure()
    legends = []
    for k, ls in history.items():
        plt.plot(range(len(ls)), ls, alpha=0.5)
        legends.append(k)
    plt.legend(legends)
    plt.savefig(filename)
    plt.close()


def plot_lr_history(lr_history, filename):
    plt.figure()
    plt.plot(range(len(lr_history)), lr_history)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["lr"])
    plt.savefig(filename)
    plt.close()


# %%


def detect_device():
    import torch
    if torch.cuda.is_available():
        return {"gpus": 1, "precision": 16}
    return {"precision": 32}


def inference(trainer, model, df, dataloader):
    prediction = torch.cat(
        trainer.predict(
            model=model,
            dataloaders=dataloader,
        )
    ).detach().cpu().numpy()
    df["target"] = prediction
    return df[["id", "target"]]


def main(CFG):
    os.chdir(CFG.dir.work_dir)

    # seed
    pl.seed_everything(CFG.general.seed)

    # device
    device_params = detect_device()

    # prepare df
    train_df = pd.read_csv(
        os.path.join(
            CFG.dir.input_dir,
            "train.csv"
        )
    ).sort_values("id")
    if CFG.general.debug:
        train_df = train_df.sample(200)
        CFG.training.epochs = 5
        CFG.training.n_fold = 3
        CFG.log.mlflow.experiment_name = "debug"

    # logger
    LOGGER = pl.loggers.MLFlowLogger(
        experiment_name=CFG.log.mlflow.experiment_name,
        save_dir=CFG.log.mlflow.save_dir,
    )

    train_texts = train_df["excerpt"].values
    train_labels = train_df["target"].values

    # train
    oof_df_ls = []

    kf = StratifiedKFold(
        CFG.training.n_fold,
        shuffle=True,
        random_state=CFG.training.shuffle_seed,
    )
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, pd.qcut(train_df["target"], CFG.training.n_fold).cat.codes)):
        print("#" * 30, f"fold: {fold}", "#" * 30)
        model = Model()

        train_dataloader = get_dataloader(
            texts=train_texts[train_idx],
            labels=train_labels[train_idx],
            transform=get_transform(data="train"),
        )
        valid_dataloader = get_dataloader(
            texts=train_texts[valid_idx],
            labels=train_labels[valid_idx],
            transform=get_transform(data="train"),
        )

        CHECKPOINT_NAME = \
            f"fold{fold}_{CFG.model.name}_""{valid_loss_epoch:.2f}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=CHECKPOINT_NAME,
            monitor='valid_loss_epoch',
            mode='min',
            save_top_k=1,
        )

        CFG.training.steps_per_epoch = len(train_dataloader)

        trainer = pl.Trainer(
            max_epochs=CFG.training.epochs,
            logger=LOGGER,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            **device_params,
        )

        trainer.fit(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        model.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        model.freeze()

        oof_dataloader = get_test_dataloader(
            texts=train_texts[valid_idx],
            transform=get_transform(data="test"),
        )

        oof_prediction = inference(
            trainer,
            model,
            train_df.iloc[valid_idx],
            oof_dataloader
        )

        oof_df_ls.append(oof_prediction)

        plot_training_curve(
            model.history, filename=f"training_curve_fold{fold}.png"
        )
        plot_lr_history(
            model.lr_history, filename=f"lr_scheduler_fold{fold}.png"
        )

    oof_df = pd.concat(oof_df_ls, axis=0).sort_values("id")
    oof_df.to_csv("oof.csv", index=False)

    validation_score = np.sqrt(mean_squared_error(
        train_df["target"], oof_df["target"])
    )
    print(f"validation score: {validation_score}")

    plot_dist(train_df["target"], oof_df["target"], filename="oof_dist.png")

    LOGGER.log_hyperparams(CFG)
    LOGGER.log_metrics({"validation_score": validation_score})
    LOGGER.experiment.log_artifact(LOGGER._run_id, __file__)
    LOGGER.experiment.log_artifact(LOGGER._run_id, "oof.csv")
    LOGGER.experiment.log_artifact(LOGGER._run_id, "oof_dist.png")


if __name__ == "__main__":
    CFG = OmegaConf.load("../config/config.yaml")
    main(CFG)

# %%
