
import numpy as np
import torch
import torch.nn as nn


def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)


class Saint(nn.Module):
    def __init__(self, d_model:int=32, nhead:int=2, d_ff:int=64, nlayers:int=2, dropout:float=0.3, max_seq_len=100, weights=None):
        '''
        nhead -> number of heads in the transformer multi attention thing.
        nhid -> the number of hidden dimension neurons in the model.
        nlayers -> how many layers we want to stack.
        '''
        super(Saint, self).__init__()
        self.src_mask = None
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=d_ff,
                                                 dropout=dropout,
                                                 activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers, norm=nn.LayerNorm(d_model))

        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=d_ff,
                                                    dropout=dropout,
                                                    activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers, norm=nn.LayerNorm(d_model))

        self.exercise_embeddings = nn.Embedding(num_embeddings=13523, embedding_dim=d_model) if weights is None else nn.Embedding.from_pretrained(weights, freeze=False)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dec_pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.prior_question_elapsed_time = nn.Linear(1, d_model, bias=False)
        self.lagtime = nn.Embedding(150, d_model)
        self.has_attempted = nn.Embedding(2, d_model)
        self.d_model = d_model

        self.generator = nn.Linear(d_model, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.prior_question_elapsed_time.weight.data.uniform_(-initrange, initrange)
        self.lagtime.weight.data.uniform_(-initrange, initrange)
        self.has_attempted.weight.data.uniform_(-initrange, initrange)
        init_weights(self.transformer_encoder)
        init_weights(self.transformer_decoder)
        init_weights(self.generator)

    def forward(self, content_id, prior_question_elapsed_time, lagtime, has_attempted, mask_src=None, mask_src_key=None):
        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        '''

        content_id = self.exercise_embeddings(content_id)
        pos = self.pos_embedding(torch.arange(0, content_id.shape[1]).to(content_id.device).unsqueeze(0).repeat(content_id.shape[0], 1))
        embedded_src = content_id + pos
        embedded_src = embedded_src.transpose(0, 1) # (S, N, E)

        _src = embedded_src * np.sqrt(self.d_model)
        memory = self.transformer_encoder(src=_src, mask=mask_src, src_key_padding_mask=mask_src_key)

        pos = self.dec_pos_embedding(torch.arange(0, content_id.shape[1]).to(content_id.device).unsqueeze(0).repeat(content_id.shape[0], 1))
        prior_question_elapsed_time = self.prior_question_elapsed_time(prior_question_elapsed_time)
        lagtime = self.lagtime(lagtime)
        has_attempted = self.has_attempted(has_attempted)
        embedded_src = pos + prior_question_elapsed_time + lagtime + has_attempted
        embedded_src = embedded_src.transpose(0, 1)

        output = self.transformer_decoder(tgt=embedded_src,
                                          memory=memory,
                                          tgt_mask=mask_src,
                                          memory_mask=mask_src,
                                          tgt_key_padding_mask=mask_src_key,
                                          memory_key_padding_mask=mask_src_key)

        output = self.generator(output)
        output = output.transpose(1, 0)
        return output
