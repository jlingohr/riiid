import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

from sklearn.metrics import auc
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.metrics.classification import ROC
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torch.optim.lr_scheduler import _LRScheduler

from datasets.riiid_dataset import load_data

class SaintPLearner(pl.LightningModule):
    def __init__(self, model, config, fold=0, dt_string=None, debug=False):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.best_loss = np.inf
        self.fold = fold
        self.save_dir = 'runs/{}/{}/{}'.format(config['model_name'], dt_string, fold)
        self.debug = debug

        self.train_roc = ROC(compute_on_step=False)
        self.val_roc = ROC(compute_on_step=False)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.src_mask = self.generate_src_mask(config['max_seq_len']) if config['use_mask'] else None

        self.noam = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def forward(self, x):
        out = self.model(x)
        return out

    def step_batch(self, batch):
        question_ids = batch['question_ids'].long()
        prior_elapsed_time = batch['prior_elapsed_time'].unsqueeze(-1)
        lagtime = batch['lagtime'].long()
        has_attempted = batch['has_attempted'].long()
        src_mask_key = batch['padded']
        target = batch['target'].long()

        # tgt, content_id, prior_question_elapsed_time, lagtime, has_attempted, mask_src=None, mask_src_key=None
        out = self.model(question_ids, prior_elapsed_time, lagtime, has_attempted, self.src_mask.to(target.device), src_mask_key)
        return out.reshape(-1, 2)

    def training_step(self, batch, batch_idx):
        target = batch['target'].long()

        logits = self.step_batch(batch)
        target = target.contiguous().view(-1)
        loss = self.criterion(logits, target)

        mask = batch['padded'].contiguous().view(-1)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = preds[~mask]

        target = target[~mask]
        self.train_roc(preds, target)
        step_acc = self.train_acc(preds, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['target'].long()

        logits = self.step_batch(batch)
        target = target.contiguous().view(-1)

        loss = self.criterion(logits, target) #TODO which target?

        mask = batch['padded'].contiguous().view(-1)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = preds[~mask]

        target = target[~mask]
        self.val_roc(preds, target)
        step_acc = self.train_acc(preds, target)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def training_epoch_end(self, outputs):
        fpr, tpr, thresholds = self.train_roc.compute()
        auc_score = auc(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
        self.log('train_auroc', auc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        acc = self.train_acc.compute()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        fpr, tpr, thresholds = self.val_roc.compute()
        auc_score = auc(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
        self.log('val_auroc', auc_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        acc = self.train_acc.compute()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if avg_loss.item() < self.best_loss:
            self.best_loss = avg_loss.item()
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, '{}/best.pth'.format(self.save_dir))
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, '{}/last.pth'.format(self.save_dir))

    def prepare_data(self):
        print("Loading datasets...")
        self.train_dataset, self.val_dataset = load_data(self.fold, self.config['max_seq_len'], self.config['train_root'], self.debug)
        print("Datasets loaded\n")

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.config['batch_size'],
                                shuffle=True,
                                drop_last=True,
                                num_workers=self.config['num_workers'])
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset,
                                batch_size=2*self.config['batch_size'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=self.config['num_workers'])
        return dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim,
                            self.config['optimizer']['type'])(self.model.parameters(),
                                                              **self.config['optimizer']['params'])

        scheduler = getattr(torch.optim.lr_scheduler,
                            self.config['scheduler']['type'])(optimizer,
                                                            **self.config['scheduler']['params'])
        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'monitor': 'val_loss'
            }

        return [optimizer], [scheduler]

    @staticmethod
    def generate_src_mask(sz):
        future_mask = np.triu(np.ones((sz, sz)), k=1).astype('bool')
        return torch.from_numpy(future_mask)
