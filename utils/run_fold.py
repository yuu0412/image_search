from utils.functions import EarlyStopping
from utils.train import training
from utils.validation import evaluation
import hydra

def run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, device, logger, cv):

    model.to(device)
    early_stopping = EarlyStopping(patience=2, verbose=True, delta=0, path=f'best_param_of_cv_{cv}.pt', trace_func=logger.info)
    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []

    for epoch in range(max_epochs):
        logger.info(f'=============== epoch:{epoch+1}/{max_epochs} ===============')
        ########################
        #        train         #
        ########################
        logger.info('------------- start of training ------------')
        train_loss = training(model, train_loader, criterion, optimizer, device, logger)
        train_losses.append(train_loss)
        # train_scores.append(train_score)

        ########################
        #      evaluation      #
        ########################
        logger.info('------------- start of evaluation ------------')
        val_loss = evaluation(model, val_loader, criterion, device, logger)
        val_losses.append(val_loss)
        # val_scores.append(val_score)

        logger.info(f'[result of epoch {epoch}/{max_epochs}]')
        logger.info(f'train_loss:{train_loss}')
        logger.info(f'val_loss:{val_loss}')

        ########################
        #     early stopping   #
        ########################
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("early stopping is adopted.")
            break

    return None