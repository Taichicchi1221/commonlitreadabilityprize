# %%
import gc
import json
import os
import pickle
import random
import re
import string
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import transformers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import mlflow

warnings.simplefilter("ignore")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


CFG = {
    "INPUT_DIR": "../input/commonlitreadabilityprize",
    "SEED": 42,

    "BATCH_SIZE": 64,

    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    "SCHEDULER": 'CosineAnnealingLR',
    "LEARNING_RATE": 1e-4,
    "MIN_LR": 1e-6,
    "WEIGHT_DECAY": 1e-6,
    "FACTOR": 0.2,  # ReduceLROnPlateau
    "PATIENCE": 4,  # ReduceLROnPlateau
    "EPS": 1e-6,  # ReduceLROnPlateau
    "T_MAX": 6,  # CosineAnnealingLR
    "T_0": 5,  # CosineAnnealingWarmRestarts

    "MODEL_NAME": "",
    "PRETRAINED": True,
    "IMAGE_SIZE_0": 224,
    "IMAGE_SIZE_1": 224,

    "EPOCHS": 6,

    "TARGET_SIZE": 1,
    "N_FOLD": 4,

    "DEBUG": False,
}

seed_everything(CFG["SEED"])


# In[5]:


train_df = pd.read_csv(Path(CFG["INPUT_DIR"], "train_labels.csv"))
test_df = pd.read_csv(Path(CFG["INPUT_DIR"], "sample_submission.csv"))

train_df["file_path"] = train_df["id"].apply(
    lambda x: Path(CFG["INPUT_DIR"], "train", f"{x[0]}/{x}.npy"))
test_df["file_path"] = test_df["id"].apply(
    lambda x: Path(CFG["INPUT_DIR"], "test", f"{x[0]}/{x}.npy"))
test_df["target"] = 0

print(f"train images = {len(train_df)}")
print(f"test images = {len(test_df)}")


# In[6]:


# DEBUG?
if CFG["DEBUG"]:
    CFG["EPOCHS"] = 3
    train_df = train_df.sample(1000).reset_index(drop=True)
    test_df = test_df.sample(1000).reset_index(drop=True)


# In[7]:


# ====================================================
# Transforms
# ====================================================

def read_image(file_path):
    image = np.load(file_path)  # (6, 273, 256)
    image = image.astype(np.float32)
    image = np.concatenate(image, axis=0).transpose(
        (1, 0))  # (1638, 256) -> (256, 1638)
    return image


def get_transform(*, data):
    if data == "train":
        return A.Compose([
            A.Resize(CFG["IMAGE_SIZE_0"], CFG["IMAGE_SIZE_1"]),
            ToTensorV2(),
        ])

    elif data == "valid":
        return A.Compose([
            A.Resize(CFG["IMAGE_SIZE_0"], CFG["IMAGE_SIZE_1"]),
            ToTensorV2(),
        ])

    elif data == "test":
        return A.Compose([
            A.Resize(CFG["IMAGE_SIZE_0"], CFG["IMAGE_SIZE_1"]),
            ToTensorV2(),
        ])


# In[8]:


# ====================================================
# scheduler
# ====================================================
def get_scheduler(optimizer):
    if CFG["SCHEDULER"] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=CFG["FACTOR"], patience=CFG["PATIENCE"], verbose=True, eps=CFG["EPS"])
    elif CFG["SCHEDULER"] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["T_MAX"], eta_min=CFG["MIN_LR"], last_epoch=-1)
    elif CFG["SCHEDULER"] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG["T_0"], T_mult=1, eta_min=CFG["MIN_LR"], last_epoch=-1)
    return scheduler


# In[9]:


# ====================================================
# Dataset
# ====================================================
class DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        file_paths,
    ):
        self.file_paths = file_paths
        self.length = len(file_paths)

    def __len__(
        self,
    ):
        return self.length

    def __getitem__(
        self,
        idx
    ):
        file_path = self.file_paths[idx]
        image = read_image(file_path)

        return image


class Dataset(DatasetBase):
    def __init__(
        self,
        file_paths,
        labels,
        transform,
    ):
        super().__init__(file_paths)
        self.labels = labels
        self.transform = transform

    def __getitem__(
        self,
        idx,
    ):
        image = super().__getitem__(idx)
        image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()

        return image, label


class TestDataset(DatasetBase):
    def __init__(
        self,
        file_paths,
        transform=None,
    ):
        super().__init__(file_paths)
        self.transform = transform

    def __getitem__(
        self,
        idx,
    ):
        image = super().__getitem__(idx)
        image = self.transform(image=image)["image"]

        return image


# In[10]:


# ====================================================
# MODEL
# ====================================================
class Model(pl.LightningModule):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, CFG["TARGET_SIZE"])

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=CFG["LEARNING_RATE"])
        scheduler = get_scheduler(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return {"loss": loss, "y_hat": y_hat, "target": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log(
            'valid_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return {"loss": loss, "y_hat": y_hat, "target": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out['loss'] for out in outputs], dim=0).mean()
        predictions = torch.sigmoid(
            torch.cat([out['y_hat'] for out in outputs], dim=0)).detach().cpu().numpy()

        self.valid_predictions = predictions
        return {'valid_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch).flatten()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        predictions = torch.sigmoid(
            torch.cat([out['y_hat'] for out in outputs], dim=0)).detach().cpu().numpy()
        self.test_predictions = predictions
        # We need to return something
        return {'dummy_item': 0}


# In[11]:


# ====================================================
# DATA MODULE
# ====================================================
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file_paths,
        valid_file_paths,
        test_file_paths,
        train_labels,
        valid_labels,
        test_labels,
        train_transform,
        valid_transform,
        test_transform,
        batch_size,
    ):
        super(DataModule, self).__init__()
        self.train_file_paths = train_file_paths
        self.valid_file_paths = valid_file_paths
        self.test_file_paths = test_file_paths
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
                self.train_file_paths,
                self.train_labels,
                self.train_transform
            )
            self.valid_dataset = Dataset(
                self.valid_file_paths,
                self.valid_labels,
                self.valid_transform,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = TestDataset(
                self.test_file_paths,
                self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False
        )


# In[12]:


skf = StratifiedKFold(CFG["N_FOLD"], shuffle=True, random_state=CFG["SEED"])

oof_predictions = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df["file_path"], train_df["target"])):
    print("#" * 30, f"fold{fold}", "#" * 30)
    train_file_paths = np.array(train_df["file_path"])[train_idx]
    valid_file_paths = np.array(train_df["file_path"])[valid_idx]
    train_labels = np.array(train_df["target"])[train_idx]
    valid_labels = np.array(train_df["target"])[valid_idx]

    test_file_paths = test_df["file_path"]

    model = Model(
        CFG["MODEL_NAME"],
        pretrained=CFG["PRETRAINED"],
    )
    dm = DataModule(
        train_file_paths=train_file_paths,
        valid_file_paths=valid_file_paths,
        test_file_paths=test_file_paths,
        train_labels=train_labels,
        valid_labels=valid_labels,
        test_labels=None,
        train_transform=get_transform(data="train"),
        valid_transform=get_transform(data="valid"),
        test_transform=get_transform(data="test"),
        batch_size=CFG["BATCH_SIZE"],
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        f"fold{fold}_{CFG['MODEL_NAME']}",
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=CFG["EPOCHS"],
        gpus=1,
        precision=16,
        deterministic=False,
        benchmark=True,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    oof_predictions[valid_idx] += model.valid_predictions
    predictions += model.test_predictions / CFG["N_FOLD"]


# In[13]:


valid_score = roc_auc_score(train_df["target"], oof_predictions)
valid_score_name = str(valid_score).split('.')[1][:4]
print(f"validation score: {valid_score}")


# In[14]:


oof_df = train_df.copy()
oof_df.rename(columns={"target": "ground_truth"}, inplace=True)
oof_df["target"] = oof_predictions
oof_df[["id", "target"]].to_csv(
    f"oof_{CFG['MODEL_NAME']}_{valid_score_name}.csv", index=False)


# In[15]:


test_df["target"] = predictions
test_df[["id", "target"]].to_csv(
    f"submission_{CFG['MODEL_NAME']}_{valid_score_name}.csv", index=False)


# In[16]:


oof_df["target"].plot.hist(bins=100)


# In[17]:


oof_df["pred_class"] = (oof_df["target"] > 0.5).astype("int")
oof_df["is_false"] = (oof_df["pred_class"] ^ oof_df["ground_truth"])
false_list = list(
    zip(
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 0), "id"].to_list(),
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 0), "ground_truth"].to_list(),
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 0), "target"].to_list(),
    )
)
random.shuffle(false_list)

W = 3
H = 7
fig = plt.figure(figsize=(30, 20))
for m, (image_id, ground_truth, target) in enumerate(false_list):
    if m == H * W:
        break
    plt.subplot(H, W, m + 1)
    plt.title(f"{image_id}, gt={ground_truth}, pred={target:.2f}")
    plt.imshow(read_image(
        f"{CFG['INPUT_DIR']}train/{image_id[0]}/{image_id}.npy"))
plt.show()


# In[18]:


false_list = list(
    zip(
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 1), "id"].to_list(),
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 1), "ground_truth"].to_list(),
        oof_df.loc[(oof_df["is_false"] == 1) & (
            oof_df["ground_truth"] == 1), "target"].to_list(),
    )
)
random.shuffle(false_list)

fig = plt.figure(figsize=(30, 20))
for m, (image_id, ground_truth, target) in enumerate(false_list):
    if m == H * W:
        break
    plt.subplot(H, W, m + 1)
    plt.title(f"{image_id}, gt={ground_truth}, pred={target:.2f}")
    plt.imshow(read_image(
        f"{CFG['INPUT_DIR']}train/{image_id[0]}/{image_id}.npy"))
plt.show()


# In[19]:


test_df["target"].plot.hist(bins=100)
