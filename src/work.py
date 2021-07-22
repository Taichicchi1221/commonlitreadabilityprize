# %%
import gc
import glob
import json
import math
import os
import pickle
import random
import re
import shutil
import string
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import transformers
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import CHECKPOINT_PAST_HPARAMS_KEYS
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader
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
    ].sort_values("id").reset_index(drop=True)
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
            tokens = self.tokenizer.encode_plus(
                text,
                truncation=True,
                padding="max_length",
                pad_to_max_length=True,
                max_length=self.tokenizer_max_length,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors="pt"
            )
            return {key: val.squeeze(0) for key, val in tokens.items()}

        return transform

    def get_collate_fn(self):
        return None

# ====================================================
# loss
# ====================================================


def get_loss(loss_name, loss_params):
    return getattr(
        torch.nn,
        loss_name
    )(**loss_params)

# ====================================================
# optimizer
# ====================================================


class Lamb(torch.optim.Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


def get_optimizer(model, optimizer_name, optimizer_params):
    if optimizer_name == "Lamb":
        return Lamb(model.parameters(), **optimizer_params)
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
            text = self.transform(text)
            return text
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
        label = label = torch.tensor(self.labels[idx]).float()
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


class MultiDropout(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        multi_dropout_rate=0.5,
        multi_dropout_num=5,
    ):
        super().__init__()

        self.dropouts = nn.ModuleList([
            nn.Dropout(multi_dropout_rate) for _ in range(multi_dropout_num)
        ])
        linears = [
            nn.Linear(in_features, out_features)
            for i in range(multi_dropout_num)
        ]
        for l in linears:
            nn.init.constant_(l.bias, -1.0)

        self.regressors = nn.ModuleList([
            linears[i] for i in range(multi_dropout_num)
        ])

    def forward(self, x):
        output = torch.stack(
            [
                regressor(dropout(x)) for regressor, dropout in zip(self.regressors, self.dropouts)
            ]
        ).mean(axis=0)
        return output


class Attention(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        return torch.sum(weights * x, dim=1)


class MeanPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.linear.bias, -1.0)

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        logits = self.linear(norm_mean_embeddings).squeeze(-1)

        return logits


class BaseModel(nn.Module):
    def __init__(
        self,
        basemodel_name: str,
        hidden_features: int,
        multi_dropout_rate: float,
        multi_dropout_num: int,
        freeze_embeddings: bool,
        model_params: dict,
    ):
        super().__init__()

        # model
        config = transformers.AutoConfig.from_pretrained(
            basemodel_name
        )
        config.update(model_params)
        config.update({"output_hidden_states": True})
        self.model = transformers.AutoModel.from_pretrained(
            basemodel_name,
            config=config,
        )

        if freeze_embeddings:
            self.model.embeddings.requires_grad_(False)

        # self.head = nn.Sequential(
        #     nn.LayerNorm(config.hidden_size),
        #     Attention(
        #         in_features=config.hidden_size,
        #         hidden_features=hidden_features,
        #     ),
        #     # nn.LayerNorm(config.hidden_size),
        #     MultiDropout(
        #         in_features=config.hidden_size,
        #         out_features=1,
        #         multi_dropout_rate=multi_dropout_rate,
        #         multi_dropout_num=multi_dropout_num,
        #     )
        # )

        self.mean_pooling = MeanPooling(config.hidden_size)

    def forward(self, x):
        last_hidden_state = self.model(**x).hidden_states[-1]
        # output = self.head(last_hidden_state)
        output = self.mean_pooling(last_hidden_state, x["attention_mask"])
        return output.squeeze(-1)


class Model(pl.LightningModule):
    def __init__(
        self,
        basemodel_name: str,
        model_params: dict,
        hidden_features: int,
        multi_dropout_rate: float,
        multi_dropout_num: int,
        freeze_embeddings: bool,
        loss_name: str,
        loss_params: dict,
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
            hidden_features=hidden_features,
            multi_dropout_rate=multi_dropout_rate,
            multi_dropout_num=multi_dropout_num,
            freeze_embeddings=freeze_embeddings,
            model_params=model_params,
        )

        # optimizer and shceduler
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

        # critetria
        self.criterion = get_loss(loss_name, loss_params)

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
        if self.scheduler_name is None:
            return {"optimizer": optimizer, }
        else:
            scheduler = get_scheduler(
                optimizer,
                scheduler_name=self.scheduler_name,
                scheduler_params=self.scheduler_params,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid_rmse",
                    "interval": self.scheduler_interval,
                }
            }

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        x, y = batch
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


def plot_lr_scheduler(lr_history, filename, steps_per_epoch, accumulate_grad_batches):
    epoch_index = [
        step for step in range(len(lr_history)) if step % (steps_per_epoch * accumulate_grad_batches) == 0
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

# ====================================================
# train fold
# ====================================================


def train_fold(
    fold,
    train_texts,
    train_labels,
    train_idx,
    valid_idx,
    logger,
    CFG
):
    print("#" * 30, f"fold: {fold}", "#" * 30)
    # device
    device_params = detect_device()

    model = Model(
        basemodel_name=CFG.model.name,
        hidden_features=CFG.model.hidden_features,
        multi_dropout_rate=CFG.model.multi_dropout_rate,
        multi_dropout_num=CFG.model.multi_dropout_num,
        freeze_embeddings=CFG.model.freeze_embeddings,
        model_params=CFG.model.params,
        loss_name=CFG.loss.name,
        loss_params=CFG.loss.params,
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

    CHECKPOINT_NAME = f"fold{fold}_{CFG.model.name}_""{epoch:02d}_{valid_rmse:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=CHECKPOINT_NAME,
        monitor='valid_rmse',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
    )

    CFG.training.steps_per_epoch = (
        len(train_dataloader) + CFG.training.accumulate_grad_batches - 1
    ) // CFG.training.accumulate_grad_batches

    trainer = pl.Trainer(
        max_epochs=CFG.training.epochs,
        logger=logger,
        benchmark=True,
        deterministic=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=CFG.training.accumulate_grad_batches,
        precision=CFG.training.precision,
        stochastic_weight_avg=CFG.training.stochastic_weight_avg,
        **device_params,
    )

    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    print(f"best model path: {checkpoint_callback.best_model_path}")

    del train_dataloader, valid_dataloader, trainer, transform_train
    gc.collect()
    torch.cuda.empty_cache()
    pl.utilities.memory.garbage_collection_cuda()

    plot_training_curve(
        model.history["train_rmse"],
        model.history["valid_rmse"],
        filename=f"training_curve_fold{fold}.png"
    )

    plot_lr_scheduler(
        model.history["lr"],
        filename=f"lr_scheduler_fold{fold}.png",
        steps_per_epoch=CFG.training.steps_per_epoch,
        accumulate_grad_batches=CFG.training.accumulate_grad_batches,
    )

    model.load_state_dict(
        torch.load(checkpoint_callback.best_model_path)["state_dict"],
    )

    model.freeze()
    model.eval()

    return model

# ====================================================
# for kaggle notebook inference
# ====================================================


def inference_main(CFG, checkpoint_paths):
    os.chdir(CFG.dir.work_dir)
    # seed
    pl.seed_everything(CFG.general.seed)

    # device
    device_params = detect_device()

    pl.seed_everything(CFG.general.seed)
    test_df = pd.read_csv(
        os.path.join(
            CFG.dir.input_dir,
            "test.csv"
        )
    )
    test_df["target"] = -1
    test_df["standard_error"] = -1
    test_df = preprocess_df(test_df)

    test_ids = test_df["id"].values
    test_texts = test_df["excerpt"].values

    trainer = pl.Trainer(
        precision=CFG.training.precision,
        logger=None,
        **device_params,
    )

    predictions_df = pd.DataFrame()

    for checkpoint_path in checkpoint_paths:

        model = Model.load_from_checkpoint(
            checkpoint_path,
            basemodel_name=CFG.model.name,
            hidden_features=CFG.model.hidden_features,
            multi_dropout_rate=CFG.model.multi_dropout_rate,
            multi_dropout_num=CFG.model.multi_dropout_num,
            freeze_embeddings=CFG.model.freeze_embeddings,
            model_params=CFG.model.params,
            loss_name=CFG.loss.name,
            loss_params=CFG.loss.params,
            optimizer_name=CFG.optimizer.name,
            optimizer_params=CFG.optimizer.params,
            scheduler_name=CFG.scheduler.name,
            scheduler_params=CFG.scheduler.params,
            scheduler_interval=CFG.scheduler.interval,
        )

        model.freeze()
        model.eval()

        prediction = inference(
            test_ids,
            test_texts,
            trainer=trainer,
            model=model,
            tokenizer_name=CFG.tokenizer.name,
            tokenizer_max_length=CFG.tokenizer.max_length,
            loader_params=CFG.loader.test,
        )

        predictions_df = pd.concat([predictions_df, prediction], axis=0)

    predictions_df = predictions_df.groupby(
        "id").mean().reset_index(drop=False)
    predictions_df.sort_values("id", inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)
    predictions_df.to_csv("submission.csv", index=False)


# ====================================================
# main
# ====================================================

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
    oof_df = pd.DataFrame()

    # fold
    kf = eval(CFG.training.splitter)(
        CFG.training.n_fold,
        shuffle=True,
        random_state=CFG.general.seed,
    )
    fold_x = train_texts
    fold_y = pd.cut(train_labels, 30).codes

    for fold, (train_idx, valid_idx) in enumerate(kf.split(fold_x, fold_y)):
        model = train_fold(
            fold,
            train_texts,
            train_labels,
            train_idx,
            valid_idx,
            logger=MLFLOW_LOGGER,
            CFG=CFG
        )

        predict_trainer = pl.Trainer(
            precision=CFG.training.precision,
            logger=None,
            callbacks=None,
            **device_params,
        )

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

        oof_df = pd.concat([oof_df, oof_prediction], axis=0)

        del predict_trainer, model, oof_ids, oof_texts
        gc.collect()
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    oof_df = oof_df.groupby("id").mean().reset_index(drop=False)
    oof_df.sort_values("id", inplace=True)
    oof_df.reset_index(drop=True, inplace=True)
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

    # inference check
    CHECKPOINT_PATHS = glob.glob(
        os.path.join(
            CFG.log.mlflow.save_dir,
            MLFLOW_LOGGER.experiment_id,
            MLFLOW_LOGGER._run_id,
            "checkpoints",
            "*.ckpt"
        )
    )

    inference_main(CFG, CHECKPOINT_PATHS)


if __name__ == "__main__":
    CFG = OmegaConf.load(
        "/workspaces/commonlitreadabilityprize/config/config.yaml"
    )
    main(CFG)


# %%
