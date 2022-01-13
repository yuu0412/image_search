import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.run_fold import run_fold
from utils.functions import init_logger, seed_torch
from utils.preprocessing import make_crop
from models.metric_net import MetricNet
from sklearn.model_selection import KFold
from datasets.jpo_dataset import JPODataset
from sklearn.preprocessing import LabelEncoder

import os
import sys
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="JPO")
def main(cfg):
    df_train = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/train.csv')
    df_test = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/test.csv')
    df_cite_v2 = pd.read_csv(hydra.utils.get_original_cwd()+'/data/input/cite_v2.csv')

    seed_torch(50)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    train_data = df_train[["gid", "path", "category"]]
    train_label = df_train["cite_gid"]

    # cite_gidはカテゴリカル変数なので、数値変数に変換する必要がある。
    encoder = LabelEncoder()
    train_label = encoder.fit_transform(train_label)
    train_label = pd.Series(train_label)

    for i, (train_idx, val_idx) in enumerate(kf.split(train_data, train_label)):
        print(f'start cv{i}...')
        train_x, val_x = train_data.iloc[train_idx], train_data.iloc[val_idx]
        train_y, val_y = train_label.iloc[train_idx], train_label.iloc[val_idx]

        train_dataset = JPODataset(train_x, train_y)
        val_dataset = JPODataset(val_x, val_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_loader.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_loader.batch_size)

        model = MetricNet(
                n_classes=df_cite_v2['gid'].nunique(),
                model_name=cfg.model.name,
                use_fc=cfg.model.use_fc,
                fc_dim=cfg.model.fc_dim,
                dropout=cfg.model.dropout,
                loss_module=cfg.model.loss_module,
                s=cfg.model.s,
                margin=cfg.model.margin,
                ls_eps=cfg.model.ls_eps,
                theta_zero=cfg.model.theta_zero,
                pretrained=cfg.model.pretrained
                )

        criterion = nn.__getattribute__(cfg.criterion.name)()
        optimizer = torch.optim.__getattribute__(cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr)
        logger = init_logger("train_log")

        max_epochs = 100 # コンソールから入力できるようにする
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, device, logger, cv=i)

if __name__ == '__main__':
    main()