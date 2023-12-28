import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ImageEncoder, TextEncoder, ProjectionHead
from .loss_fn import cross_entropy_loss, infonce_ns_loss, infonce_loss


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
        loss_fn: str = "cross_entropy",
        label_type: str = "game",
    ):
        super().__init__()

        self.img_model_name = img_model_name
        self.text_model_name = text_model_name
        self.temperature = temperature
        self.loss_fn = loss_fn
        self.label_type = label_type

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

        # Calculating the Loss
        if self.loss_fn == "cross_entropy":
            loss = cross_entropy_loss(img_embeddings, text_embeddings, self.temperature)
        elif self.loss_fn == "infonce_ns_loss":
            category_labels = (
                batch["game_label"]
                if self.label_type == "game"
                else batch["genre_label"]
            )
            loss = infonce_ns_loss(
                img_embeddings, text_embeddings, category_labels, self.temperature
            )
        else:
            loss = infonce_loss(img_embeddings, text_embeddings, self.temperature)

        return {
            "loss": loss.mean(),
            "img_embeddings": img_embeddings,
            "text_embeddings": text_embeddings,
        }
