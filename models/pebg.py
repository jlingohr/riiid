import numpy as np
import torch
import torch.nn as nn


class PEBG(nn.Module):
    def __init__(self, num_questions, num_tags, diff_feat_size, dim_size, hidden_size):
        super(PEBG, self).__init__()
        self.question_embedding = nn.Embedding(num_questions, dim_size)
        self.skill_embedding = nn.Embedding(num_tags, dim_size)
        self.difficulty_embedding = nn.Linear(diff_feat_size, dim_size)
        self.pnn = PNN(dim_size, hidden_size)
        self.Q = self.question_embedding.weight
        self.S = self.skill_embedding.weight.t()
        
    def forward(self, batch):
        questions, question_skill_targets, difficulty_feats = batch
        
        # compute average skill features        
        mu_skill = torch.vstack([self.skill_embedding(row.nonzero()).mean(axis=0) for row in question_skill_targets])
        
        # get question vertex features for batch
        q = self.question_embedding(questions)
        
        # get difficulty project
        a = self.difficulty_embedding(difficulty_feats)

        # Run through product layers
        e, p = self.pnn([q, mu_skill, a])
        
        return e, p
    
    def R_hat(self):
        # Question-Skill
        return torch.matmul(self.question_embedding.weight, self.skill_embedding.weight.t())
    
    def Q_hat(self):
        # Question-Question similarity
        return torch.matmul(self.question_embedding.weight, self.question_embedding.weight.t())
    
    def S_hat(self):
        # Skill-skill similarity
        return torch.matmul(self.skill_embedding.weight, self.skill_embedding.weight.t())