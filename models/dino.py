

import torch
import torch.nn as nn



dino_models = {
    "small": ("dinov2_vits14_reg_lc", 384, 14),
    "base": ("dinov2_vitb14_reg_lc", 768, 14),
    "large": ("dinov2_vitl14_reg_lc", 1024, 14),
}

class DinoVitFeatureExtractorFinetune(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """

    def __init__(
        self,
        image_backbone="small",
        last_n_feat=1,
        path=None,
        
    ):
        super().__init__()
        assert (
            image_backbone in dino_models.keys()
        ), f"DinoVitFeatureExtractor is only available for {dino_models.keys()}"
        model_name, embed_dim, patch_size = dino_models[image_backbone]
        print(f"image_backbone: {image_backbone}, model_name, {model_name} embed_dim: {embed_dim}, patch_size: {patch_size}  ")

        self.last_n_feat = last_n_feat
        self.embed_dim = embed_dim * self.last_n_feat

   
        path = "facebookresearch/dinov2"
        source = 'github'
        encoder = torch.hub.load(path, model_name, source=source)
        self.encoder = encoder.backbone

        self.encoder.eval()
        self.patch_size = patch_size


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size, _, height, width = x.size()
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        output = self.encoder.get_intermediate_layers(x, n=self.last_n_feat, return_class_token=True)
        x, _ = output[0]
        x = x.transpose(1, 2).view(batch_size, self.embed_dim, f_height, f_width)
        return x
    

