import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    
    def __init__(self, temperature=0.07, use_cosine_similarity=True): 
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Normalize features
        if self.use_cosine_similarity:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        
        similarity_matrix = torch.matmul(representations, representations.T)
        
        #Create mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        

        #Create labels
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(z1.device)
                
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss


class SimplifiedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07, margin=0.5):
        super(SimplifiedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, z1, z2, labels):
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(z1, z2, dim=1)

        # Scale by temperature
        similarity = similarity / self.temperature

        # Compute loss
        pos_loss = -similarity
        neg_loss = torch.clamp(similarity - self.margin, min=0.0)

        loss = labels * pos_loss + (1 - labels) * neg_loss

        return loss.mean()


class WeightedContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07, margin=0.5, adversarial_weight=3.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.adversarial_weight = adversarial_weight

    def forward(self, z1, z2, labels, is_adversarial=None):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute cosine similarity & compute loss
        similarity = F.cosine_similarity(z1, z2, dim=1)
        similarity = similarity / self.temperature
        neg_loss = torch.clamp(similarity - self.margin, min=0.0)

        # Weighted by labels
        loss = labels * pos_loss + (1 - labels) * neg_loss

        if is_adversarial is not None:
            # higher weight for adv negative pairs (hard negatives)
            weights = torch.ones_like(loss)
            adversarial_negatives = (~labels.bool()) & is_adversarial.bool()
            weights[adversarial_negatives] = self.adversarial_weight
            loss = loss * weights

        return loss.mean()


class CombinedLoss(nn.Module):
    
    def __init__(self, lambda_contrastive=1.0, lambda_flow=0.5, lambda_change=0.5,
                 temperature=0.07, use_simplified=False, use_weighted=False,
                 adversarial_weight=3.0):
        super(CombinedLoss, self).__init__()

        self.lambda_contrastive = lambda_contrastive
        self.lambda_flow = lambda_flow
        self.lambda_change = lambda_change

        # Contrastive loss
        if use_weighted:
            self.contrastive_loss = WeightedContrastiveLoss(
                temperature=temperature,
                adversarial_weight=adversarial_weight
            )
        elif use_simplified:
            self.contrastive_loss = SimplifiedContrastiveLoss(temperature=temperature)
        else:
            self.contrastive_loss = NTXentLoss(temperature=temperature)

        self.use_simplified = use_simplified
        self.use_weighted = use_weighted
        
        self.flow_loss = nn.L1Loss()
        self.change_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, z1, z2, labels=None, is_adversarial=None, flow_pred=None,
                flow_gt=None, change_pred=None, change_gt=None):
        loss_dict = {}

        if self.use_weighted or self.use_simplified:
            assert labels is not None, "Labels required for simplified/weighted contrastive loss"
            if self.use_weighted:
                loss_contrastive = self.contrastive_loss(z1, z2, labels, is_adversarial)
            else:
                loss_contrastive = self.contrastive_loss(z1, z2, labels)
        else:
            loss_contrastive = self.contrastive_loss(z1, z2)

        loss_dict['contrastive'] = loss_contrastive.item()
        total_loss = self.lambda_contrastive * loss_contrastive
        
        # flow loss
        if flow_pred is not None and flow_gt is not None:
            loss_flow = self.flow_loss(flow_pred, flow_gt)
            loss_dict['flow'] = loss_flow.item()
            total_loss += self.lambda_flow * loss_flow
        
        # change loss
        if change_pred is not None and change_gt is not None:
            loss_change = self.change_loss(change_pred, change_gt)
            loss_dict['change'] = loss_change.item()
            total_loss += self.lambda_change * loss_change
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing Contrastive Loss Functions\n")
    
    batch_size = 8
    projection_dim = 128
    
    # Create dummy features
    z1 = torch.randn(batch_size, projection_dim)
    z2 = torch.randn(batch_size, projection_dim)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    print(f"Batch size: {batch_size}")
    print(f"Projection dim: {projection_dim}")
    print(f"Labels: {labels}")

    print("\nTesting NT-Xent Loss")
    
    ntxent = NTXentLoss(temperature=0.07)
    loss = ntxent(z1, z2)
    print(f"NT-Xent Loss: {loss.item():.4f}")
    
    print("\nTesting Simplified Contrastive Loss")
    
    simplified = SimplifiedContrastiveLoss(temperature=0.07)
    loss = simplified(z1, z2, labels)
    print(f"Simplified Loss: {loss.item():.4f}")

    print("\nTesting Combined Loss")
    
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
    
    print(f"\n All tests passed!")
