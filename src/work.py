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
import torchmetrics
from tqdm import tqdm

warnings.simplefilter("ignore")

# %%

# ====================================================
# preprocess
# ====================================================


def preprocess_df(df):
    df["target"] = df["target"].astype(np.float32)
    df = df.loc[
        ~((df["target"] == 0) & (df["standard_error"] == 0))
    ].reset_index(drop=True).sort_values("id")
    return df

# ====================================================
# metric
# ====================================================


class RMSE(torchmetrics.Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state(
            "sum_squared_errors",
            torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_observations",
            torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)


# ====================================================
# transform
# ====================================================


class Transform():
    def __init__(self, data, tokenizer_name, tokenizer_max_length):
        self.data = data
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_length = tokenizer_max_length

        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        config = transformers.AutoConfig.from_pretrained(
            self.tokenizer_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            config=config,
        )
        return tokenizer

    def get_transform_fn(self):
        def transform(text):
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.tokenizer_max_length
            )
            return tokens

        return transform

    def get_collate_fn(self):
        return transformers.DataCollatorWithPadding(self.tokenizer)


# ====================================================
# optimizer
# ====================================================


def get_optimizer(model, optimizer_name, optimizer_params):
    return getattr(
        torch.optim,
        optimizer_name
    )(model.parameters(), **optimizer_params)

# ====================================================
# scheduler
# ====================================================


def get_scheduler(optimizer, scheduler_name, scheduler_params):
    return getattr(
        torch.optim.lr_scheduler,
        scheduler_name
    )(optimizer, **scheduler_params)


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
            return self.transform(text)
        return {"text": text}


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
        label = self.labels[idx]
        text.update({"label": label})
        return text


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


class BaseModel(nn.Module):
    def __init__(
        self,
        basemodel_name: str,
        multisample_dropout: int,
        multisample_dropout_rate: float,
        model_params,
    ):
        super().__init__()

        # model
        config = transformers.AutoConfig.from_pretrained(
            basemodel_name
        )
        config.update(model_params)
        self.model = transformers.AutoModel.from_pretrained(
            basemodel_name,
            config=config,
        )
        self.middle_layer_norm = nn.LayerNorm(config.hidden_size)
        self.middle_linear = nn.Linear(
            config.hidden_size, config.hidden_size // 2
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size // 2)
        self.dropouts = nn.ModuleList([
            nn.Dropout(multisample_dropout_rate) for _ in range(multisample_dropout)
        ])
        self.regressors = nn.ModuleList([
            nn.Linear(config.hidden_size // 2, 1) for _ in range(multisample_dropout)
        ])

    def forward(self, x):
        output = self.model(**x)[1]
        output = self.middle_layer_norm(output)
        output = F.relu(self.middle_linear(output))
        output = self.layer_norm(output)
        logits = torch.stack(
            [
                regressor(dropout(output)) for regressor, dropout in zip(self.regressors, self.dropouts)
            ]
        ).mean(axis=0)
        return logits.flatten()


class Model(pl.LightningModule):
    def __init__(
        self,
        basemodel_name: str,
        multisample_dropout: int,
        multisample_dropout_rate: float,
        model_params: dict,
        optimizer_name: str,
        optimizer_params: dict,
        scheduler_name: str,
        scheduler_params: dict,
        scheduler_interval: str,
    ):
        super().__init__()

        # model
        self.model = BaseModel(
            basemodel_name=basemodel_name,
            multisample_dropout=multisample_dropout,
            multisample_dropout_rate=multisample_dropout_rate,
            model_params=model_params,
        )

        # optimizer and shceduler
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

        # critetria
        self.criterion = torch.nn.MSELoss()

        # metrics
        self.train_rmse = RMSE()
        self.valid_rmse = RMSE()

        # init model training histories
        self.history = {
            "train_rmse": [],
            "valid_rmse": [],
            "lr": []
        }

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            optimizer_params=self.optimizer_params,
        )
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=self.scheduler_name,
            scheduler_params=self.scheduler_params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
            }
        }

    @staticmethod
    def split_batch(batch):
        x = {}
        y = {}
        for k in batch.keys():
            if k == "labels":
                y = batch[k]
            else:
                x.update({k: batch[k]})
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_rmse(y_hat, y)
        self.log(
            name="train_rmse",
            value=self.train_rmse,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        self.history["lr"].append(
            self.optimizers(False).param_groups[0]["lr"]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.valid_rmse(y_hat, y)
        self.log(
            name="valid_rmse",
            value=self.valid_rmse,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.history["train_rmse"].append(
            self.train_rmse.compute().detach().cpu().numpy()
        )
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.history["valid_rmse"].append(
            self.valid_rmse.compute().detach().cpu().numpy()
        )
        return super().on_validation_epoch_end()

# ====================================================
# dataloader
# ====================================================


def get_dataloader(
    texts,
    labels,
    transform,
    collate_fn,
    loader_params,
):
    ds = Dataset(texts, labels, transform)
    dl = DataLoader(ds, collate_fn=collate_fn, **loader_params)

    return dl


def get_test_dataloader(
    texts,
    transform,
    collate_fn,
    loader_params,
):
    ds = TestDataset(texts, transform)
    dl = DataLoader(ds, collate_fn=collate_fn, **loader_params)

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


def plot_training_curve(train_history, valid_history, filename):
    plt.figure()
    legends = []
    plt.plot(range(len(train_history)), train_history,
             marker=".", color="skyblue")
    legends.append("train")
    plt.plot(range(len(valid_history)), valid_history,
             marker=".", color="orange")
    legends.append("valid")
    plt.legend(legends)
    plt.savefig(filename)
    plt.close()


def plot_lr_scheduler(lr_history, filename, steps_per_epoch):
    epoch_index = [
        step for step in range(len(lr_history)) if step % steps_per_epoch == 0
    ]
    plt.figure()
    plt.plot(range(len(lr_history)), lr_history)
    plt.plot(
        [i for i in range(len(lr_history)) if i in epoch_index],
        [lr_history[i] for i in range(len(lr_history)) if i in epoch_index],
        color="orange",
        linestyle="None",
        marker="D"
    )
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["lr", "epoch"])
    plt.savefig(filename)
    plt.close()


# ====================================================
# util
# ====================================================

def detect_device():
    import torch
    if torch.cuda.is_available():
        return {"gpus": 1}
    return {"gpus": None}

# ====================================================
# inference
# ====================================================


def inference(
    test_ids,
    test_texts,
    trainer,
    model,
    tokenizer_name,
    tokenizer_max_length,
    loader_params,
):
    print(f"inference check: model.model.training={model.model.training}")
    transform = Transform(
        data="test",
        tokenizer_name=tokenizer_name,
        tokenizer_max_length=tokenizer_max_length
    )
    dataloader = get_test_dataloader(
        texts=test_texts,
        transform=transform.get_transform_fn(),
        collate_fn=transform.get_collate_fn(),
        loader_params=loader_params,
    )
    prediction = torch.cat(
        trainer.predict(
            model=model,
            dataloaders=dataloader,
        )
    ).detach().cpu().numpy()
    df = pd.DataFrame()
    df["id"] = test_ids
    df["target"] = prediction
    return df


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
    )
    train_df = preprocess_df(train_df)

    if CFG.general.debug:
        train_df = train_df.sample(200)
        CFG.training.epochs = 5
        CFG.training.n_fold = 3
        CFG.log.mlflow.experiment_name = "debug"

    # logger
    MLFLOW_LOGGER = pl.loggers.MLFlowLogger(
        experiment_name=CFG.log.mlflow.experiment_name,
        save_dir=CFG.log.mlflow.save_dir,
    )

    train_texts = train_df["excerpt"].values
    train_labels = train_df["target"].values

    # train
    oof_df_ls = []

    kf = eval(CFG.training.splitter)(
        CFG.training.n_fold,
        shuffle=True,
        random_state=CFG.training.shuffle_seed,
    )
    for fold, (train_idx, valid_idx) in enumerate(
        kf.split(
            train_df, pd.cut(train_df["target"], 5 *
                             CFG.training.n_fold).cat.codes
        )
    ):
        print("#" * 30, f"fold: {fold}", "#" * 30)

        model = Model(
            basemodel_name=CFG.model.name,
            multisample_dropout=CFG.model.multisample_dropout,
            multisample_dropout_rate=CFG.model.multisample_dropout_rate,
            model_params=CFG.model.params,
            optimizer_name=CFG.optimizer.name,
            optimizer_params=CFG.optimizer.params,
            scheduler_name=CFG.scheduler.name,
            scheduler_params=CFG.scheduler.params,
            scheduler_interval=CFG.scheduler.interval,
        )

        transform_train = Transform(
            data="train",
            tokenizer_name=CFG.tokenizer.name,
            tokenizer_max_length=CFG.tokenizer.max_length,
        )

        train_dataloader = get_dataloader(
            texts=train_texts[train_idx],
            labels=train_labels[train_idx],
            transform=transform_train.get_transform_fn(),
            collate_fn=transform_train.get_collate_fn(),
            loader_params=CFG.loader.train,
        )
        valid_dataloader = get_dataloader(
            texts=train_texts[valid_idx],
            labels=train_labels[valid_idx],
            transform=transform_train.get_transform_fn(),
            collate_fn=transform_train.get_collate_fn(),
            loader_params=CFG.loader.train,
        )

        CHECKPOINT_NAME = \
            f"fold{fold}_{CFG.model.name}_""{epoch:02d}_{valid_rmse:.2f}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=CHECKPOINT_NAME,
            monitor='valid_rmse',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
        )

        CFG.training.steps_per_epoch = len(train_dataloader)

        trainer = pl.Trainer(
            max_epochs=CFG.training.epochs,
            logger=MLFLOW_LOGGER,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            precision=CFG.training.precision,
            stochastic_weight_avg=CFG.training.stochastic_weight_avg,
            **device_params,
        )

        trainer.fit(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        plot_training_curve(
            model.history["train_rmse"],
            model.history["valid_rmse"],
            filename=f"training_curve_fold{fold}.png"
        )

        plot_lr_scheduler(
            model.history["lr"],
            filename=f"lr_scheduler_fold{fold}.png",
            steps_per_epoch=CFG.training.steps_per_epoch
        )

        del trainer, model, train_dataloader, valid_dataloader
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

        predict_trainer = pl.Trainer(
            precision=CFG.training.precision,
            logger=None,
            **device_params,
        )
        model = Model.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
            basemodel_name=CFG.model.name,
            multisample_dropout=CFG.model.multisample_dropout,
            multisample_dropout_rate=CFG.model.multisample_dropout_rate,
            model_params=CFG.model.params,
            optimizer_name=None,
            optimizer_params=None,
            scheduler_name=None,
            scheduler_params=None,
            scheduler_interval=None,
        )
        model.eval()
        model.freeze()

        oof_ids = train_df["id"].values[valid_idx]
        oof_texts = train_df["excerpt"].values[valid_idx]
        oof_prediction = inference(
            oof_ids,
            oof_texts,
            trainer=predict_trainer,
            model=model,
            tokenizer_name=CFG.tokenizer.name,
            tokenizer_max_length=CFG.tokenizer.max_length,
            loader_params=CFG.loader.test,
        )

        oof_df_ls.append(oof_prediction)

        del predict_trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    oof_df = pd.concat(oof_df_ls, axis=0).sort_values("id")
    oof_df.to_csv("oof.csv", index=False)

    validation_score = mean_squared_error(
        train_df["target"], oof_df["target"], squared=False
    )
    print(f"validation score: {validation_score}")

    plot_dist(
        train_df["target"],
        oof_df["target"],
        filename="oof_dist.png"
    )

    MLFLOW_LOGGER.log_hyperparams(CFG)
    open("config.yaml", "w").write(OmegaConf.to_yaml(CFG))

    MLFLOW_LOGGER.log_metrics({"validation_score": validation_score})
    if globals().get("__file__"):
        MLFLOW_LOGGER.experiment.log_artifact(MLFLOW_LOGGER._run_id, __file__)
    MLFLOW_LOGGER.experiment.log_artifact(MLFLOW_LOGGER._run_id, "config.yaml")
    MLFLOW_LOGGER.experiment.log_artifact(MLFLOW_LOGGER._run_id, "oof.csv")
    MLFLOW_LOGGER.experiment.log_artifact(
        MLFLOW_LOGGER._run_id, "oof_dist.png")
    for fold in range(CFG.training.n_fold):
        MLFLOW_LOGGER.experiment.log_artifact(
            MLFLOW_LOGGER._run_id, f"training_curve_fold{fold}.png"
        )
        MLFLOW_LOGGER.experiment.log_artifact(
            MLFLOW_LOGGER._run_id, f"lr_scheduler_fold{fold}.png"
        )


if __name__ == "__main__":
    CFG = OmegaConf.load(
        "/workspaces/commonlitreadabilityprize/config/config.yaml"
    )
    main(CFG)

# %%
