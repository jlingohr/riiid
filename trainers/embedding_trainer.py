import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F


class PEBGTrainer:
    def __init__(self, model, device):
        self.device = device
        self.pebg = model
        
    def train(self, num_epochs, loader):
        self.pebg.to(self.device)
        self.pebg.train()
        optimizer = Adam(self.pebg.parameters(), lr=0.001)
        
        skill_skill_matrix = torch.tensor(loader.dataset.skill_skill_matrix()).to(self.device)
        best_loss = float('inf')
        losses = {
            "L1": [],
            "L2": [],
            "L3": [],
            "L4": [],
            "combined": []
        }
        
        bce = nn.BCEWithLogitsLoss()
        
        for epoch in tqdm(range(num_epochs)):
            epoch_l1 = 0.0
            epoch_l2 = 0.0
            epoch_l3 = 0.0
            epoch_l4 = 0.0
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                questions = batch['question_id'].to(self.device)
                question_skill_targets = batch['question_skill_target'].float().to(self.device)
                question_question_targets = batch['question_question_target'].float().to(self.device)
                difficulty_feats = batch['difficulty_feats'].float().to(self.device)
                auxiliary_targets = batch['auxiliary_target'].float().to(self.device)
                
                optimizer.zero_grad()
                
                # Compute approximations
                e, p = self.pebg([questions, question_skill_targets, difficulty_feats])
                
                # Question-Skill
                R_hat = self.pebg.R_hat()[questions]

                # Question-Question similarity
                RQ_hat = self.pebg.Q_hat()[questions]

                # Skill-skill similarity
                RS_hat = self.pebg.S_hat()
                
                # compute losses
                L1 = bce(R_hat.view(-1), question_skill_targets.view(-1)).mean()
                L2 = bce(RQ_hat.view(-1), question_question_targets.view(-1)).mean()
                L3 = bce(RS_hat.view(-1), skill_skill_matrix.view(-1)).mean()
                L4 = F.mse_loss(p, auxiliary_targets.unsqueeze(-1)).mean()
                
                loss = L1 + L2 + L3 + L4
                loss.backward()
                optimizer.step()
                
                epoch_l1 += L1.item()
                epoch_l2 += L2.item()
                epoch_l3 += L3.item()
                epoch_l4 += L4.item()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            if epoch_loss < best_loss:
                torch.save({'model_state_dict': self.pebg.state_dict()}, 'best.pth')
            torch.save({'model_state_dict': self.pebg.state_dict()}, 'last.pth')
            
            losses['L1'].append(epoch_l1 / len(loader))
            losses['L2'].append(epoch_l2 / len(loader))
            losses['L3'].append(epoch_l3 / len(loader))
            losses['L4'].append(epoch_l4 / len(loader))
            losses['combined'].append(epoch_loss)
        return losses