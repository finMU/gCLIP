import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import transformers
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(
        self, model_name: str, use_pretrained: bool = True, is_trainable: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.is_trainable = is_trainable

        # img encoer init
        self.model = timm.create_model(
            model_name, num_classes=0, global_pool="avg", pretrained=use_pretrained
        ) 
        # TODO : 이 부분도 timm 없이, transformers만으로도 사용 가능한 것 같습니다

        if not self.is_trainable:
            for parameter in self.model.parameters():
                parameter.requires_grad = self.is_trainable

    def forward(self, img: torch.Tensor):
        return self.model(img)
    
    
class TextEncoder(nn.Module):
    def __init__(
        self, model_name: str, use_pretrained: bool = True, is_trainable: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.is_trainable = is_trainable
        self.cls_token_idx = 0

        if use_pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            raise NotImplementedError
            # TODO : use_pretrained가 아니라면, pretraining을 하는 기능을 이 부분에 구현해야함

        if not self.is_trainable:
            for parameter in self.model.parameters():
                parameter.requires_grad = self.is_trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.cls_token_idx, :]
    
    
class ProjectionHead(nn.Module):
    """
    ref:  https://github.com/h-albert-lee/G-CLIP/blob/master/modules.py

    TODO: img encoder에 layer_norm 적절한지?
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x