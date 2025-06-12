import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.dino import DinoVitFeatureExtractorFinetune
from models.blocks import Mlp

# Global config (consider moving this to a config file or initialization script)
torch.backends.cuda.matmul.allow_tf32 = True  # For GPUs >= Ampere and PyTorch >= 1.12

def set_requires_grad(module: nn.Module, requires_grad: bool):
    """Utility function to set requires_grad for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = requires_grad

class DIPNet(nn.Module):
    """
    DIPNet integrates a backbone (DINO, CLIP, or MAE) with an MLP projector and
    a cross-attention mechanism to produce segmentation outputs.
    """
    def __init__(self,
                 enc_embed_dim: int = 384,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 mlp_ratio: int = 7,
                 image_backbone: str = "small",
                 num_classes: int = 1000,
                 do_training: int = 1,
                 interpolate: str = 'nearest',
                 beta: float = 0.07,
                 layers_to_finetune: int = 3,
                 do_distractors: bool = True,
                 n_distractor_samples: int = 7,
                 mlp_out_features: int = 384,
                 image_size: int = 224,
                 backbone_type: str = "dino"):
        super(DIPNet, self).__init__()
        self.num_classes = num_classes
        self.do_training = do_training == 1
        self.interpolate = interpolate
        self.beta = beta
        self.do_distractors = do_distractors
        self.n_distractor_samples = n_distractor_samples
        self.mlp_out_features = mlp_out_features

        # Adjust embedding dimension for specific backbones
        enc_embed_dim = 384
        if image_backbone == "base":
            enc_embed_dim = 768

        self.initialize_weights()
        self.projector = Mlp(in_features=enc_embed_dim,
                             out_features=mlp_out_features,
                             hidden_features=int(enc_embed_dim * mlp_ratio))
        # breakpoint()
        self.enc_norm = norm_layer(mlp_out_features)
        self.backbone_type = backbone_type
        self.patch_size = None  # will be set in _setup_backbone

        # Initialize backbone and set up layers to fine-tune
        self.feats_extractor = self._setup_backbone(
            backbone_type, image_backbone, layers_to_finetune)

        self.num_patches = (image_size // self.patch_size) ** 2

    def initialize_weights(self):
        """Initialize weights for Linear and LayerNorm layers."""
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _setup_backbone(self, backbone_type: str, image_backbone: str,
                        layers_to_finetune: int) -> nn.Module:
        """Initializes and returns the feature extractor backbone based on the specified type."""
        total_layers = 12  # This can be parameterized based on the actual backbone

        if backbone_type == "dino":
            self.patch_size = 14
            extractor = DinoVitFeatureExtractorFinetune(
                image_backbone=image_backbone)
            # Freeze entire encoder
            set_requires_grad(extractor.encoder, False)
            # Unfreeze last few layers
            for i in range(total_layers - layers_to_finetune, total_layers):
                set_requires_grad(extractor.encoder.blocks[i], True)
            set_requires_grad(extractor.encoder.norm, True)
            return extractor

        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    @staticmethod
    def patchify_gt(gt: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Patchify ground truth images.
        Args:
            gt (torch.Tensor): Input tensor of shape (bs, c, h, w).
            patch_size (int): Size of each patch.
        Returns:
            torch.Tensor: Patchified tensor.
        """
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        return gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)

    @staticmethod
    def cross_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute cross attention.
        Args:
            q (torch.Tensor): Query tensor of shape (bs, num_patches, d_k).
            k (torch.Tensor): Key tensor of shape (bs, num_patches, d_k).
            v (torch.Tensor): Value tensor of shape (bs, num_patches, label_dim).
            beta (float): Temperature parameter.
        Returns:
            torch.Tensor: Result of attention operation.
        """
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (q @ k.transpose(-2, -1)) / beta
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def _augment_distractors(self, label: torch.Tensor, feat_ref_post: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Generates distractors the reference"""
        bs, N, _ = label.shape
        device = feat_ref_post.device
        total_samples = N * (self.n_distractor_samples + 1)

        distractors_label = torch.empty((bs, total_samples, self.num_classes), device=device, dtype=label.dtype)
        distractors_feat_ref_post = torch.empty((bs, total_samples, feat_ref_post.shape[-1]),
                                        device=device, dtype=feat_ref_post.dtype)
        # Copy original positive samples
        distractors_label[:, :N, :] = label
        distractors_feat_ref_post[:, :N, :] = feat_ref_post

        for i in range(self.n_distractor_samples):
            perm = torch.randperm(bs, device=device)
            start_idx = (i + 1) * N
            end_idx = start_idx + N
            distractors_label[:, start_idx:end_idx, :] = label[perm, :, :]
            distractors_feat_ref_post[:, start_idx:end_idx, :] = feat_ref_post[perm, :, :]

        return distractors_label, distractors_feat_ref_post

    def forward(self, img_target: torch.Tensor, img_ref: torch.Tensor, labels_ref: torch.Tensor) -> torch.Tensor:
        bs, _, h, w = img_target.shape
        eval_spatial_resolution = int(math.sqrt(self.num_patches))

        # Extract features based on backbone type
       
        feat_target = self.feats_extractor(img_target)
        feat_ref = self.feats_extractor(img_ref)
        # Flatten features if necessary
        if feat_target.dim() == 4:
            feat_target = feat_target.flatten(2).transpose(1, 2)
            feat_ref = feat_ref.flatten(2).transpose(1, 2)
        # breakpoint()
        # Process features through projector if training
        if self.do_training:
            feat_target_post = self.enc_norm(self.projector(feat_target))
            feat_ref_post = self.enc_norm(self.projector(feat_ref))
        else:
            feat_target_post = feat_target
            feat_ref_post = feat_ref

        # Compute ground truth labels from patchified GT
        patchified_gts = self.patchify_gt(labels_ref, self.patch_size).long()
        one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=self.num_classes).float()
        label = one_hot_patch_gt.mean(dim=3).reshape(bs, -1, self.num_classes)

        # Generate distractor samples if enabled
        if self.do_distractors:
            label_distractors, feat_ref_post_distractors = self._augment_distractors(label, feat_ref_post)
        else:
            label_distractors, feat_ref_post_distractors = label, feat_ref_post

        # Compute cross attention
        label_hat = self.cross_attention(feat_target_post, feat_ref_post_distractors, label_distractors, self.beta)
        label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, self.num_classes)\
                             .permute(0, 3, 1, 2)
        resized_label_hats = F.interpolate(label_hat.float(), size=(h, w), mode=self.interpolate)
        return resized_label_hats
