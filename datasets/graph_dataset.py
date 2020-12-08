import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
    def __init__(self, questions_path, num_tags=188):
        # initialize adjacency matrix and associated matrices
        self.question_skill_graph = self.build_adjacency(questions_path, num_tags)
        self.question_question_graph = self.question_skill_graph @ self.question_skill_graph.T
        self.question_question_graph[self.question_question_graph > 0] = 1
        self.skill_skill_graph = self.question_skill_graph.T @ self.question_skill_graph
        self.skill_skill_graph[self.skill_skill_graph > 0] = 1
        self.question_features = self.build_features(questions_path)
        
    def __len__(self):
        return len(self.question_skill_graph)
    
    def __getitem__(self, idx):
        question_skill_target = self.question_skill_graph[idx]
        question_question_target = self.question_question_graph[idx]
        difficulty_feats = self.question_features[idx, :-1]
        auxiliary_target = self.question_features[idx, -1]
        
        sample = {
            'question_id': idx,
            'question_skill_target': question_skill_target,
            'question_question_target': question_question_target,
            'difficulty_feats': difficulty_feats,
            'auxiliary_target': auxiliary_target
        }
        
        return sample
        
    def build_adjacency(self, path, num_tags):
        questions = self.load_data(path)
        
        M,N = len(questions), num_tags
        graph = np.zeros((M,N))
        
        for rowIdx, row in questions.iterrows():
            for tag in row.tags:
                graph[row.question_id][tag] = 1
        
        return graph
    
    def skill_skill_matrix(self):
        return self.skill_skill_graph
    
    def build_features(self, path):
        questions = self.load_data(path)
        questions['num_answers'] = questions.part.apply(lambda x: 0 if x == 2 else 1)
        questions['is_reading'] = questions.part.apply(lambda x: 0 if x < 5 else 1)
        parts = pd.get_dummies(questions.part, prefix='part')
        attributes = pd.concat([parts, questions.num_answers, questions.is_reading, questions.difficulty], axis=1)
        return attributes.to_numpy()
    
    def load_data(self, path):
        questions = pd.read_csv(path)
        questions['tags'] = questions['tags'].apply(lambda ts: [int(x) for x in ts.split() if x != 'nan'])
        return questions