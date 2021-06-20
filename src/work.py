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
import warnings
from functools import partial
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from yaml import tokens
import seaborn as sns
import torch
import transformers
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.simplefilter("ignore")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


def get_config():
    with open("../config/config.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# ====================================================
# Transform
# ====================================================


class Transform():
    def __init__(self, data, CFG):
        self.CFG = CFG
        self.data = data
        self.tokenizer = self.get_tokenizer()

    def clean_text(self, text):
        '''
        Converts all text to lower case, Removes special charecters, emojis and multiple spaces
        text - Sentence that needs to be cleaned
        '''
        text = ''.join([k for k in text if k not in string.punctuation])
        text = re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
        text = re.sub(' +', ' ', text)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        return text

    def get_tokenizer(self):
        config = transformers.AutoConfig.from_pretrained(
            self.CFG["MODEL_NAME"]
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.CFG["MODEL_NAME"],
            config=config,
        )
        return tokenizer

    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.CFG["TRANSFORM"]["max_length"],
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

    def __call__(self, text):
        if self.CFG["TRANSFORM"]["text_cleaning"]:
            text = self.clean_text(text)
        tokens = self.tokenize(text)

        return tokens

# ====================================================
# Optimizer, Scheduler
# ====================================================


def get_optimizer(model, CFG):
    optimizer_name = CFG["OPTIMIZER"]["name"]
    optimizer_params = CFG["OPTIMIZER"][f"params_{CFG['OPTIMIZER']['name']}"]
    optimizer = getattr(
        torch.optim,
        optimizer_name
    )(model.parameters(), **optimizer_params)
    return optimizer


def get_scheduler(optimizer, CFG):
    scheduler_name = CFG["SCHEDULER"]["name"]
    scheduler_params = CFG["SCHEDULER"][f"params_{CFG['SCHEDULER']['name']}"]
    scheduler = getattr(
        torch.optim.lr_scheduler,
        scheduler_name
    )(optimizer, **scheduler_params)
    return scheduler


# ====================================================
# Dataset
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
        if self.transform:
            text = self.transform(self.texts[idx])
            text = {
                k: torch.tensor(v, dtype=torch.long) for k, v in text.items()
            }

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
# MODEL
# ====================================================
class Model(pl.LightningModule):
    def __init__(
        self,
        CFG
    ):
        super().__init__()
        self.CFG = CFG
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            CFG["MODEL_NAME"],
            num_labels=1,
        )

    def forward(self, x):
        output = self.model(**x)
        return output.logits.flatten()

    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.CFG)
        scheduler = get_scheduler(optimizer, self.CFG)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.sqrt(F.mse_loss(y_hat, y))
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.sqrt(F.mse_loss(y_hat, y))
        self.log(
            'valid_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

# ====================================================
# DATA MODULE
# ====================================================


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_texts,
        valid_texts,
        test_texts,
        train_labels,
        valid_labels,
        test_labels,
        train_transform,
        valid_transform,
        test_transform,
        batch_size,
    ):
        super(DataModule, self).__init__()
        self.train_texts = train_texts
        self.valid_texts = valid_texts
        self.test_texts = test_texts
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(
                self.train_texts,
                self.train_labels,
                self.train_transform
            )
            self.valid_dataset = Dataset(
                self.valid_texts,
                self.valid_labels,
                self.valid_transform,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = TestDataset(
                self.test_texts,
                self.test_transform
            )

    def setup_oof(self):
        self.oof_dataset = TestDataset(
            self.valid_texts,
            self.valid_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False,
            drop_last=False
        )

    def oof_dataloader(self):
        return DataLoader(
            self.oof_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=False,
            drop_last=False
        )

# %%


def detect_device():
    import torch
    if torch.cuda.is_available():
        return {"gpus": 1, "precision": 16}
    return {"precision": 32}


CONFIG_PATH = "../config/config.yaml"


@hydra.main(config_path=CONFIG_PATH)
def main(CFG):
    # device
    device_params = detect_device()

    # prepare df
    train_df = pd.read_csv(
        os.path.join(
            hydra.utils.get_original_cwd(),
            CFG["IO"]["INPUT_DIR"],
            "train.csv"
        )
    )
    if CFG["BASE"]["DEBUG"]:
        train_df = train_df.sample(200)
        CFG["TRAIN"]["epochs"] = 1

    train_texts = train_df["excerpt"].values
    train_labels = train_df["target"].values

    # train
    oof = np.zeros(train_labels.shape)
    kf = KFold(
        CFG["TRAIN"]["n_fold"],
        shuffle=True,
        random_state=CFG["TRAIN"]["shuffle_seed"],
    )
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        print("#" * 30, f"fold: {fold}", "#" * 30)
        model = Model(CFG)
        dm = DataModule(
            train_texts=train_texts[train_idx],
            valid_texts=train_texts[valid_idx],
            test_texts=None,
            train_labels=train_labels[train_idx],
            valid_labels=train_labels[valid_idx],
            test_labels=None,
            train_transform=Transform("train", CFG),
            valid_transform=Transform("valid", CFG),
            test_transform=Transform("test", CFG),
            batch_size=CFG["TRAIN"]["batch_size"]
        )

        trainer = pl.Trainer(
            **device_params,
            max_epochs=CFG["TRAIN"]["epochs"]
        )

        trainer.fit(model, datamodule=dm)

        model.freeze()

        dm.setup_oof()
        prediction = torch.cat(
            trainer.predict(model, dm.oof_dataloader())
        ).detach().cpu().numpy()
        oof[valid_idx] = prediction

    validation_score = np.sqrt(mean_squared_error(train_df["target"], oof))
    plt.figure()
    plt.hist(train_df["target"], alpha=0.5, bins=100)
    plt.hist(oof, alpha=0.5, bins=100)
    plt.legend(["ground-truth", "oof"])
    plt.savefig("oof_plot.png")
    plt.close()
    print(f"validation score: {validation_score}")
    f = open(f"val_score: {validation_score}")
    f.close()

    # copy this script to current directory
    shutil.copyfile(__file__, os.path.join(".", os.path.basename(__file__)))


if __name__ == "__main__":
    main()

# %%
