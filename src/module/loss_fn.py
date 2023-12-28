import torch
import torch.nn.functional as F


def cross_entropy_loss(img_embeddings, text_embeddings, temperature):
    """
    Apply cross entropy loss(normal) between image and text embeddings.
    이미지-텍스트 유사성뿐만 아니라 이미지-이미지 및 텍스트-텍스트 유사성도 고려

    Parameters:
    - image_embeddings: Tensor of shape (batch_size, embedding_size) containing image embeddings.
    - text_embeddings: Tensor of shape (batch_size, embedding_size) containing text embeddings.
    - temperature: A temperature scaling factor for controlling the separation of the feature space.

    Returns:
    - The cross entropy loss
    """
    # Calculating the Loss
    logits = (text_embeddings @ img_embeddings.T) / temperature
    imgs_similarity = img_embeddings @ img_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T

    targets = F.softmax(
        (imgs_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    texts_loss = F.cross_entropy(logits, targets, reduction="none")
    imgs_loss = F.cross_entropy(logits.T, targets.T, reduction="none")
    loss = (imgs_loss + texts_loss) / 2.0
    return loss


def infonce_loss(img_embeddings, text_embeddings, temperature):
    """
    Apply infonce loss between image and text embeddings.
    이미지와 텍스트 간의 유사성만을 이용하여 cross-entropy 손실을 계산

    Parameters:
    - img_embeddings: Tensor of shape (batch_size, embedding_size) containing image embeddings.
    - text_embeddings: Tensor of shape (batch_size, embedding_size) containing text embeddings.
    - temperature: A temperature scaling factor for controlling the separation of the feature space.

    Returns:
    - The infonce loss
    """
    # Normalize the embeddings
    img_embeddings = F.normalize(img_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    # Compute the similarity matrix
    similarity_matrix = torch.matmul(img_embeddings, text_embeddings.t())
    # Scale by the temperature
    similarity_matrix /= temperature
    # Labels for each entry in the batch
    labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
    # Compute the loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def infonce_ns_loss(img_embeddings, text_embeddings, category_labels, temperature):
    """
    Apply category-based negative sampling for the InfoNCE loss between image and text embeddings.

    Parameters:
    - img_embeddings: Tensor of shape (batch_size, embedding_size) containing image embeddings.
    - text_embeddings: Tensor of shape (batch_size, embedding_size) containing text embeddings.
    - category_labels: Tensor of shape (batch_size,) containing the category labels for each sample.
    - temperature: A temperature scaling factor for controlling the separation of the feature space.

    Returns:
    - The modified InfoNCE loss with category-based negative sampling.
    """
    # 이미지 임베딩과 텍스트 임베딩 사이의 dot-product 계산
    logits = torch.matmul(text_embeddings, img_embeddings.T) / temperature
    # 대각선(자기자신)에 대한 마스크 생성
    diagonal_mask = torch.eye(logits.size(0), device=logits.device)
    # category-based negative sampling 기반으로 mask 생성
    categories = category_labels.unsqueeze(1)
    positive_mask = torch.eq(categories, categories.T).float()
    # 대각선을 0으로 설정하여 자기 자신을 제외
    positive_mask = positive_mask * (1 - diagonal_mask)
    # InfoNCE loss 계산
    # 각 샘플에 대한 softmax를 적용하고, positive pair만 고려
    loss = -torch.log_softmax(logits, dim=1) * positive_mask
    loss = loss.sum(dim=1) / positive_mask.sum(dim=1)  # 평균 손실 계산
    return loss.mean()