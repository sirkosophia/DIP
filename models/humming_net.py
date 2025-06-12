import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
import torch.nn.functional as F
from models.dipnet import DIPNet



def strip_prefix_from_state_dict(state_dict, prefix="feats_extractor.model."):
    stripped_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            stripped_dict[new_key] = v
        else:
            stripped_dict[k] = v  # keep as is
    return stripped_dict

    
class DIP(DIPNet):
    def __init__(self, **kwargs):
        super(DIP, self).__init__(**kwargs)

    def forward(self, x):
        feat_target= self.feats_extractor(x)
        if feat_target.dim() == 4:
            feat_target = feat_target.flatten(2).transpose(1, 2)
        feat_target_post = feat_target

        return feat_target_post
    
class Humming_net(nn.Module):
    def __init__(self, model_name, model_params, model_weights, embedding_size=384, out_dim=None):
        super(Humming_net, self).__init__()
        print(f"Model name: {model_name}")
        print(f"Model params: {model_params}")
        print(f"Model weights: {model_weights}")
        print(f"Embedding size: {embedding_size}")
       
        self.model = eval(model_name)(**model_params)
        
        if model_weights is not None:
            checkpoint = torch.load(model_weights, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                checkpoint = checkpoint['model']
                
            checkpoint = strip_prefix_from_state_dict(checkpoint, prefix="module.")
            self.model.load_state_dict(checkpoint, strict=False)
        self.embedding_size = embedding_size
        self.patch_size = 14
        self.model_name = model_name

    @torch.no_grad()
    def forward(self, x):
        x_dip = self.model(x)
        return x_dip
