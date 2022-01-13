import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import fbeta_score

import numpy as np
from tqdm import tqdm


from sklearn.metrics import fbeta_score

def training(model, train_loader, criterion, optimizer, device, logger):

    losses = []
    model.train()

    for n, batch in enumerate(train_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n % 10 == 0:
            logger.info(f'[{n}/{len(train_loader)}]: loss:{sum(losses) / len(losses)}')

        del images, labels

    loss_average = sum(losses) / len(losses)

    return loss_average