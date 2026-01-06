"""
Task 5: SimSaC Contrastive Model

Extends TAMPAR's SimSaC with a projection head for contrastive learning.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps backbone features to lower-dimensional space for contrastive loss.
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        """
        Initialize projection head.
        
        Args:
            input_dim: Dimension of backbone features (e.g., 512 for VGG pyramid)
            hidden_dim: Hidden layer dimension
            output_dim: Projection space dimension
        """
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Project features.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Projected features [batch_size, output_dim]
        """
        return self.projection(x)


class SimSaCContrastive(nn.Module):
    """
    SimSaC model extended with contrastive learning.
    
    Wraps TAMPAR's SimSaC and adds a projection head.
    """
    
    def __init__(self, simsac_model, projection_dim=128, freeze_backbone=True):
        """
        Initialize SimSaC contrastive model.
        
        Args:
            simsac_model: Pre-trained SimSaC model from TAMPAR
            projection_dim: Dimension of projection space
            freeze_backbone: Whether to freeze backbone initially
        """
        super(SimSaCContrastive, self).__init__()
        
        self.simsac = simsac_model
        self.freeze_backbone = freeze_backbone
        
        # Determine backbone feature dimension
        # For ResNet50, it's typically 2048
        # We'll infer it from the model
        self.feature_dim = self._get_feature_dim()
        
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=512,
            output_dim=projection_dim
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _get_feature_dim(self):
        """Infer feature dimension from backbone."""
        # VGG pyramid typically outputs 512-dim features from the deepest level
        return 512
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        print("Freezing backbone parameters...")
        
        # Freeze feature extraction backbone
        if hasattr(self.simsac, 'backbone'):
            for param in self.simsac.backbone.parameters():
                param.requires_grad = False
        
        # Optionally freeze flow and change decoders
        if hasattr(self.simsac, 'flow_decoder'):
            for param in self.simsac.flow_decoder.parameters():
                param.requires_grad = False
        
        if hasattr(self.simsac, 'change_decoder'):
            for param in self.simsac.change_decoder.parameters():
                param.requires_grad = False
        
        print("✓ Backbone frozen")
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        print("Unfreezing all parameters...")
        
        for param in self.simsac.parameters():
            param.requires_grad = True
        
        self.freeze_backbone = False
        print("✓ All parameters unfrozen")
    
    def extract_features(self, img):
        """
        Extract features from backbone.
        
        Args:
            img: Input image [batch_size, 3, H, W]
        
        Returns:
            features: Backbone features [batch_size, feature_dim]
        """
        # SimSaC uses a pyramid for feature extraction
        if hasattr(self.simsac, 'pyramid'):
            # Extract features from the pyramid
            features = self.simsac.pyramid(img)
            
            # Get the deepest feature level (usually level_5 or level_6)
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Take deepest level
            
            # Global average pooling if spatial dimensions exist
            if len(features.shape) == 4:  # [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)  # [B, C]
            
            return features
        else:
            raise NotImplementedError("SimSaC pyramid not found")
    
    def forward(self, img1, img2, return_features=False):
        """
        Forward pass for contrastive learning.
        
        Args:
            img1: First image [batch_size, 3, H, W]
            img2: Second image [batch_size, 3, H, W]
            return_features: If True, return backbone features too
        
        Returns:
            z1: Projected features for img1 [batch_size, projection_dim]
            z2: Projected features for img2 [batch_size, projection_dim]
            (optional) features1, features2: Backbone features
        """
        # Extract features
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        
        # Project to contrastive space
        z1 = self.projection_head(features1)
        z2 = self.projection_head(features2)
        
        if return_features:
            return z1, z2, features1, features2
        else:
            return z1, z2
    
    def get_trainable_parameters(self):
        """Get parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_params(self):
        """Print statistics about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")


def load_simsac_pretrained(weights_path, device='cuda'):
    """
    Load pre-trained SimSaC model from TAMPAR.
    
    Args:
        weights_path: Path to SimSaC weights (.pth file)
        device: Device to load model on
    
    Returns:
        simsac_model: Loaded SimSaC model
    """
    print(f"\nLoading SimSaC from: {weights_path}")
    
    try:
        # Import TAMPAR's SimSaC
        import sys
        sys.path.insert(0, "/content/tampar")
        from src.simsac.models.our_models.SimSaC import SimSaC_Model
        
        # Initialize model with TAMPAR's actual parameters
        simsac = SimSaC_Model(
            evaluation=True,  # Set to evaluation mode
            pyramid_type='VGG',  # VGG pyramid (3x3 kernels)
            md=4,  # Maximum displacement for correlation
            dense_connection=True,
            consensus_network=False,
            cyclic_consistency=False,  # Not needed for fine-tuning
            decoder_inputs='corr_flow_feat',
            num_class=2,
            use_pac=False,  # No PAC layers in this checkpoint
            batch_norm=True,
            iterative_refinement=False,
            refinement_at_all_levels=False,
            refinement_at_adaptive_reso=True,
            upfeat_channels=2,
            vpr_candidates=False,
            div=1.0
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            simsac.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            simsac.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'net' in checkpoint:
            simsac.load_state_dict(checkpoint['net'], strict=False)
        else:
            simsac.load_state_dict(checkpoint, strict=False)
        
        simsac = simsac.to(device)
        simsac.eval()
        
        print("✓ SimSaC loaded successfully")
        return simsac
        
    except Exception as e:
        print(f"✗ Error loading SimSaC: {e}")
        print("\nPlease adjust the import path and loading logic")
        print("based on your TAMPAR repository structure.")
        import traceback
        traceback.print_exc()
        raise



def create_simsac_contrastive(weights_path, projection_dim=128, 
                              freeze_backbone=True, device='cuda'):
    """
    Create SimSaC contrastive model.
    
    Args:
        weights_path: Path to pre-trained SimSaC weights
        projection_dim: Projection head output dimension
        freeze_backbone: Whether to freeze backbone initially
        device: Device to use
    
    Returns:
        model: SimSaCContrastive model ready for training
    """
    print(f"\n{'='*70}")
    print("Creating SimSaC Contrastive Model")
    print(f"{'='*70}")
    
    # Load pre-trained SimSaC
    simsac = load_simsac_pretrained(weights_path, device)
    
    # Wrap with contrastive learning
    model = SimSaCContrastive(
        simsac_model=simsac,
        projection_dim=projection_dim,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    
    # Print model info
    model.print_trainable_params()
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing SimSaC Contrastive Model\n")
    
    # Create dummy SimSaC (since we don't have actual weights here)
    class DummySimSaC(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        def extract_features(self, x):
            feat = self.backbone(x)
            return feat.view(feat.size(0), -1)
    
    dummy_simsac = DummySimSaC()
    
    # Create contrastive model
    model = SimSaCContrastive(
        simsac_model=dummy_simsac,
        projection_dim=128,
        freeze_backbone=True
    )
    
    # Test forward pass
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 256, 256)
    img2 = torch.randn(batch_size, 3, 256, 256)
    
    z1, z2 = model(img1, img2)
    
    print(f"Input shape: {img1.shape}")
    print(f"Projection output shape: {z1.shape}")
    print(f"Expected: [batch_size, projection_dim] = [{batch_size}, 128]")
    
    assert z1.shape == (batch_size, 128), f"Unexpected shape: {z1.shape}"
    assert z2.shape == (batch_size, 128), f"Unexpected shape: {z2.shape}"
    
    model.print_trainable_params()
    
    print(f"\n✓ Model test passed!")