import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ImageEncoder, TextEncoder, ProjectionHead
from ..module.loss_fn import cross_entropy_loss, infonce_ns_loss, infonce_loss


class CLIP(nn.Module):
    def __init__(
        self,
        img_model_name: str,
        text_model_name: str,
        temperature: float,
        img_embedding: int,
        text_embedding: int,
        projection_dim: int,
        dropout: float,
        is_trainable: bool = True,
        use_pretrained: bool = True,
        loss_fn : str = 'cross_entropy'
    ):
        super().__init__()

        self.img_model_name = img_model_name
        self.text_model_name = text_model_name
        self.temperature = temperature
        self.loss_fn = loss_fn

        self.img_encoder = ImageEncoder(img_model_name, use_pretrained, is_trainable)
        self.text_encoder = TextEncoder(text_model_name, use_pretrained, is_trainable)

        self.img_projection = ProjectionHead(
            embedding_dim=img_embedding, projection_dim=projection_dim, dropout=dropout
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding, projection_dim=projection_dim, dropout=dropout
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        # img & text features from encoder
        img_features = self.img_encoder(batch["image"])
        text_features = self.text_encoder(batch["input_ids"], batch["attention_mask"])

        # img & text embedding from projection head
        img_embeddings = self.img_projection(img_features)
        text_embeddings = self.text_projection(text_features)

        if self.loss_fn == 'cross_entropy':
            loss = cross_entropy_loss(img_embeddings, text_embeddings, self.temperature)
        elif self.loss_fn == 'infonce_ns_loss':
            # TODO : batch['category_label'] 구현 후 해당 부분 구현하기
            raise NotImplementedError("아직 dataset에 category_label이 구현되지 않았습니다")
        else:
            loss = infonce_loss(img_embeddings, text_embeddings, self.temperature)

        return {
            "loss": loss.mean(),
            "img_embeddings": img_embeddings,
            "text_embeddings": text_embeddings,
        }
