from sklearn.metrics import fbeta_score
import torch
import numpy as np

def evaluation(model, val_loader, criterion, device, logger):
    losses = []
    model.eval()
    for n, batch in enumerate(val_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

    loss_average = sum(losses) / len(losses)
    
    return loss_average