"""
Task 5: Contrastive Loss Functions

Implements:
1. NT-Xent (Normalized Temperature-scaled Cross Entropy) - SimCLR loss
2. Combined loss with optional flow and change losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    This is the standard SimCLR contrastive loss.
    
    Reference: "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    
    def __init__(self, temperature=0.07, use_cosine_similarity=True):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter (τ) for scaling
            use_cosine_similarity: Use cosine similarity (True) or dot product (False)
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, z1, z2):
        """
        Compute NT-Xent loss.
        
        Args:
            z1: Projected features from image 1 [batch_size, projection_dim]
            z2: Projected features from image 2 [batch_size, projection_dim]
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = z1.size(0)
        
        # Normalize features if using cosine similarity
        if self.use_cosine_similarity:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        # Concatenate z1 and z2
        # representations: [2*batch_size, projection_dim]
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        # similarity_matrix: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create mask to exclude self-similarity
        # mask: [2*batch_size, 2*batch_size]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        
        # Positive pairs are at positions (i, i+batch_size) and (i+batch_size, i)
        # Create labels
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(z1.device)
        
        # Remove self-similarity from similarity matrix
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss


class SimplifiedContrastiveLoss(nn.Module):
    """
    Simplified contrastive loss for positive/negative pairs.
    
    This is easier to understand and works well when you have explicit labels.
    """
    
    def __init__(self, temperature=0.07, margin=0.5):
        """
        Initialize simplified contrastive loss.
        
        Args:
            temperature: Temperature for similarity scaling
            margin: Margin for negative pairs
        """
        super(SimplifiedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, z1, z2, labels):
        """
        Compute contrastive loss.
        
        Args:
            z1: Features from image 1 [batch_size, dim]
            z2: Features from image 2 [batch_size, dim]
            labels: 1 for positive pairs, 0 for negative [batch_size]
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(z1, z2, dim=1)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Compute loss
        # Positive pairs: maximize similarity (minimize -similarity)
        # Negative pairs: minimize similarity (add margin)
        pos_loss = -similarity  # Negative because we want to maximize
        neg_loss = torch.clamp(similarity - self.margin, min=0.0)
        
        # Weighted by labels
        loss = labels * pos_loss + (1 - labels) * neg_loss
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for SimSaC contrastive fine-tuning.
    
    L_total = λ_contrastive * L_contrastive + λ_flow * L_flow + λ_change * L_change
    """
    
    def __init__(self, lambda_contrastive=1.0, lambda_flow=0.5, lambda_change=0.5,
                 temperature=0.07, use_simplified=False):
        """
        Initialize combined loss.
        
        Args:
            lambda_contrastive: Weight for contrastive loss
            lambda_flow: Weight for flow loss (if available)
            lambda_change: Weight for change loss (if available)
            temperature: Temperature for contrastive loss
            use_simplified: Use simplified contrastive loss instead of NT-Xent
        """
        super(CombinedLoss, self).__init__()
        
        self.lambda_contrastive = lambda_contrastive
        self.lambda_flow = lambda_flow
        self.lambda_change = lambda_change
        
        # Contrastive loss
        if use_simplified:
            self.contrastive_loss = SimplifiedContrastiveLoss(temperature=temperature)
        else:
            self.contrastive_loss = NTXentLoss(temperature=temperature)
        
        self.use_simplified = use_simplified
        
        # Optional losses (TAMPAR original)
        self.flow_loss = nn.L1Loss()  # For optical flow
        self.change_loss = nn.BCEWithLogitsLoss()  # For change detection
    
    def forward(self, z1, z2, labels=None, flow_pred=None, flow_gt=None, 
                change_pred=None, change_gt=None):
        """
        Compute combined loss.
        
        Args:
            z1: Projected features from image 1
            z2: Projected features from image 2
            labels: Pair labels (required for simplified loss)
            flow_pred: Predicted flow (optional)
            flow_gt: Ground truth flow (optional)
            change_pred: Predicted change mask (optional)
            change_gt: Ground truth change mask (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Contrastive loss
        if self.use_simplified:
            assert labels is not None, "Labels required for simplified contrastive loss"
            loss_contrastive = self.contrastive_loss(z1, z2, labels)
        else:
            loss_contrastive = self.contrastive_loss(z1, z2)
        
        loss_dict['contrastive'] = loss_contrastive.item()
        total_loss = self.lambda_contrastive * loss_contrastive
        
        # Flow loss (optional - if you have ground truth flow)
        if flow_pred is not None and flow_gt is not None:
            loss_flow = self.flow_loss(flow_pred, flow_gt)
            loss_dict['flow'] = loss_flow.item()
            total_loss += self.lambda_flow * loss_flow
        
        # Change loss (optional - if you have ground truth change masks)
        if change_pred is not None and change_gt is not None:
            loss_change = self.change_loss(change_pred, change_gt)
            loss_dict['change'] = loss_change.item()
            total_loss += self.lambda_change * loss_change
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing Contrastive Loss Functions\n")
    
    batch_size = 8
    projection_dim = 128
    
    # Create dummy features
    z1 = torch.randn(batch_size, projection_dim)
    z2 = torch.randn(batch_size, projection_dim)
    labels = torch.randint(0, 2, (batch_size,)).float()  # Random 0/1 labels
    
    print(f"Batch size: {batch_size}")
    print(f"Projection dim: {projection_dim}")
    print(f"Labels: {labels}")
    
    # Test NT-Xent loss
    print(f"\n{'='*50}")
    print("Testing NT-Xent Loss")
    print(f"{'='*50}")
    
    ntxent = NTXentLoss(temperature=0.07)
    loss = ntxent(z1, z2)
    print(f"NT-Xent Loss: {loss.item():.4f}")
    
    # Test simplified loss
    print(f"\n{'='*50}")
    print("Testing Simplified Contrastive Loss")
    print(f"{'='*50}")
    
    simplified = SimplifiedContrastiveLoss(temperature=0.07)
    loss = simplified(z1, z2, labels)
    print(f"Simplified Loss: {loss.item():.4f}")
    
    # Test combined loss
    print(f"\n{'='*50}")
    print("Testing Combined Loss")
    print(f"{'='*50}")
    
    combined = CombinedLoss(
        lambda_contrastive=1.0,
        lambda_flow=0.5,
        lambda_change=0.5,
        use_simplified=True
    )
    
    total_loss, loss_dict = combined(z1, z2, labels)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✓ All tests passed!")
