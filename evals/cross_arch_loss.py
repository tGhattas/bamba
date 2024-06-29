

from torch import nn
import torch

class EmbeddingProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.projection(x)

class AttentionProjectionLoss(nn.Module):

    def __init__(self):
        super(AttentionProjectionLoss, self).__init__()
        pass

    def forward(self, student_outputs, teacher_attention_head, *args, **kwargs):
       pass
    