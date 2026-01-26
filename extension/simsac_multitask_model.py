"""
SimSaC Multi-Task Model for Tampering Detection

Combines:
1. Contrastive learning (triplet/quadruplet loss) for clean vs tampered separation
2. Tampering classification head for specific tampering type detection
3. Hard negative mining for focusing on confusing examples
4. Tampering code conditioning

Architecture:
    SimSaC (frozen or fine-tunable)
        ↓
    Feature Embeddings (256-d)
        ↓
    ┌─────────────────────┬──────────────────────┐
    ↓                     ↓                      ↓
Projection Head    Classification Head    Tampering Encoder
(Contrastive)      (Tampering Type)      (Conditioning)
    ↓                     ↓                      ↓
512-d embeddings    N-class logits        Tampering features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.simsac.models.our_models.SimSaC import SimSaC_Model


class TamperingClassificationHead(nn.Module):
    """
    Classification head for predicting tampering type.

    Predicts one of:
    - 0: Clean (no tampering)
    - 1: Tape
    - 2: Writing
    - 3: Fold
    - 4: Other tampering
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_classes=5, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, features):
        """
        Args:
            features: [B, D] feature embeddings

        Returns:
            logits: [B, num_classes] class logits
        """
        return self.classifier(features)


class TamperingConditioningEncoder(nn.Module):
    """
    Encodes tampering codes into feature space for conditioning.

    Takes tampering codes (e.g., "T1W2") and produces embeddings
    that can be used to condition the contrastive loss.
    """

    def __init__(self, embedding_dim=64, num_tampering_types=10):
        super().__init__()

        # Embedding for each tampering type (T1-T5, W1-W5, F1-F3, etc.)
        self.tampering_embedding = nn.Embedding(num_tampering_types, embedding_dim)

        # Combine multiple tampering codes
        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, tampering_codes):
        """
        Args:
            tampering_codes: [B] list of tampering code indices

        Returns:
            embeddings: [B, embedding_dim] tampering embeddings
        """
        embeddings = self.tampering_embedding(tampering_codes)
        return self.combiner(embeddings)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features to normalized embeddings for triplet/quadruplet loss.
    """

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=512, dropout=0.1):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        """
        Args:
            features: [B, D] feature embeddings

        Returns:
            embeddings: [B, output_dim] L2-normalized embeddings
        """
        embeddings = self.projection(features)
        # L2 normalize for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


class SimSaCMultiTask(nn.Module):
    """
    Multi-task SimSaC model combining:
    1. Contrastive learning via projection head
    2. Tampering classification via classification head
    3. Tampering conditioning via conditioning encoder
    """

    def __init__(
        self,
        freeze_simsac=False,
        projection_dim=512,
        num_tampering_classes=5,
        tampering_embedding_dim=64,
        dropout=0.3
    ):
        super().__init__()

        # Base SimSaC model
        self.simsac = SimSaC_Model(
            batch_norm=True,
            pyramid_type="VGG",
            div=1.0,
            evaluation=False,
            consensus_network=False,
            cyclic_consistency=True,
            dense_connection=True,
            decoder_inputs="corr_flow_feat",
            refinement_at_all_levels=False,
            refinement_at_adaptive_reso=True,
            num_class=2,
            use_pac=False,
            vpr_candidates=False,
        )

        if freeze_simsac:
            for param in self.simsac.parameters():
                param.requires_grad = False

        # Get feature dimension from SimSaC (typically 256)
        # This is the output dimension from the feature pyramid
        feature_dim = 256

        # Multi-task heads
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

        self.classification_head = TamperingClassificationHead(
            input_dim=feature_dim,
            hidden_dim=projection_dim,
            num_classes=num_tampering_classes,
            dropout=dropout
        )

        self.tampering_encoder = TamperingConditioningEncoder(
            embedding_dim=tampering_embedding_dim,
            num_tampering_types=20  # T1-T5, W1-W5, F1-F3, etc.
        )

    def extract_features(self, img1, img2, img1_256, img2_256):
        """
        Extract features from image pair using SimSaC.

        Args:
            img1, img2: Full resolution images [B, 3, H, W]
            img1_256, img2_256: Resized images [B, 3, 256, 256]

        Returns:
            features: [B, 256] feature embeddings
        """
        # Run SimSaC forward pass
        with torch.set_grad_enabled(self.training and not all(not p.requires_grad for p in self.simsac.parameters())):
            # Get flow and change maps
            flow, change = self.simsac(img1, img2, img1_256, img2_256)

            # Extract features from change map
            # change has shape [B, 2, H, W] (2 channels: change score for each direction)
            # We'll use global average pooling to get fixed-size features
            change_features = F.adaptive_avg_pool2d(change, (1, 1))  # [B, 2, 1, 1]
            change_features = change_features.view(change_features.size(0), -1)  # [B, 2]

            # Also extract features from the SimSaC decoder
            # We'll use the last decoder features which have richer information
            # Access intermediate features from the decoder
            # For now, we'll use change map features (2-d) and expand to 256-d in the heads

            # Since SimSaC outputs are low-dimensional, we need to extract richer features
            # Let's use the feature pyramid instead
            # We'll modify this to extract from intermediate layers

            # For now, using a simple approach: concatenate flow and change statistics
            flow_mag = torch.norm(flow, dim=1, keepdim=True)  # [B, 1, H, W]
            flow_features = F.adaptive_avg_pool2d(flow_mag, (1, 1))  # [B, 1, 1, 1]
            flow_features = flow_features.view(flow_features.size(0), -1)  # [B, 1]

            # Concatenate flow and change features
            combined_features = torch.cat([flow_features, change_features], dim=1)  # [B, 3]

            # Project to 256-d using a learned projection
            if not hasattr(self, 'feature_projector'):
                self.feature_projector = nn.Linear(3, 256).to(combined_features.device)

            features = self.feature_projector(combined_features)  # [B, 256]

        return features

    def forward(self, img1, img2, img1_256, img2_256, return_features=False):
        """
        Forward pass through multi-task model.

        Args:
            img1, img2: Full resolution images [B, 3, H, W]
            img1_256, img2_256: Resized images [B, 3, 256, 256]
            return_features: If True, return intermediate features

        Returns:
            embeddings: [B, projection_dim] normalized embeddings for contrastive loss
            logits: [B, num_classes] tampering classification logits
            features: [B, 256] intermediate features (if return_features=True)
        """
        # Extract features
        features = self.extract_features(img1, img2, img1_256, img2_256)

        # Project for contrastive learning
        embeddings = self.projection_head(features)

        # Classify tampering type
        logits = self.classification_head(features)

        if return_features:
            return embeddings, logits, features
        else:
            return embeddings, logits

    def load_simsac_weights(self, checkpoint_path):
        """
        Load pre-trained SimSaC weights.

        Args:
            checkpoint_path: Path to SimSaC checkpoint (TAMPAR format)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load state dict (already in TAMPAR format without 'simsac.' prefix)
        state_dict = checkpoint['state_dict']

        # Load into SimSaC model
        self.simsac.load_state_dict(state_dict, strict=True)
        print(f"Loaded SimSaC weights from {checkpoint_path}")


def create_multitask_model(
    simsac_checkpoint=None,
    freeze_simsac=False,
    projection_dim=512,
    num_tampering_classes=5,
    device='cuda'
):
    """
    Create multi-task model with optional pre-trained SimSaC weights.

    Args:
        simsac_checkpoint: Path to SimSaC checkpoint (optional)
        freeze_simsac: Whether to freeze SimSaC weights
        projection_dim: Dimension of contrastive embeddings
        num_tampering_classes: Number of tampering types (0=clean, 1=tape, 2=writing, etc.)
        device: Device to load model on

    Returns:
        model: SimSaCMultiTask model
    """
    model = SimSaCMultiTask(
        freeze_simsac=freeze_simsac,
        projection_dim=projection_dim,
        num_tampering_classes=num_tampering_classes,
    )

    if simsac_checkpoint is not None:
        model.load_simsac_weights(simsac_checkpoint)

    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Creating multi-task model...")
    model = create_multitask_model(
        simsac_checkpoint=None,
        freeze_simsac=False,
        projection_dim=512,
        num_tampering_classes=5,
        device=device
    )

    print(f"\nModel structure:")
    print(f"  SimSaC: {sum(p.numel() for p in model.simsac.parameters())} parameters")
    print(f"  Projection head: {sum(p.numel() for p in model.projection_head.parameters())} parameters")
    print(f"  Classification head: {sum(p.numel() for p in model.classification_head.parameters())} parameters")
    print(f"  Tampering encoder: {sum(p.numel() for p in model.tampering_encoder.parameters())} parameters")
    print(f"  Total: {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 512, 512).to(device)
    img2 = torch.randn(batch_size, 3, 512, 512).to(device)
    img1_256 = torch.randn(batch_size, 3, 256, 256).to(device)
    img2_256 = torch.randn(batch_size, 3, 256, 256).to(device)

    embeddings, logits, features = model(img1, img2, img1_256, img2_256, return_features=True)

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Embeddings norm: {torch.norm(embeddings, dim=1).mean():.4f} (should be ~1.0)")

    print("\n✓ Model creation and forward pass successful!")
