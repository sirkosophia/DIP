# Code from: https://github.com/vpariza/open-hummingbird-eval

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule
from types import MethodType

class FeatureExtractorSimple(torch.nn.Module):
    def __init__(self, vit_model, ftr_extr_fn, eval_spatial_resolution=14, d_model=768):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.ftr_extr_fn = ftr_extr_fn
    
    def forward_features(self, imgs):
        return self.ftr_extr_fn(self.model, imgs)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, vit_model, eval_spatial_resolution=14, d_model=768):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
    

    def freeze_feature_extractor(self, unfreeze_layers=[]):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break

    def get_intermediate_layer_feats(self, imgs, feat="k", layer_num=-1):
        bs, c, h, w = imgs.shape
        imgs = imgs.reshape(bs, c, h, w)
        ## hook to get the intermediate layers
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.model._modules["blocks"][layer_num]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        self.model(imgs)
        attentions = self.model.get_last_selfattention(imgs)
        # Scaling factor
        average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
        temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
        normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
        # cls_attentions = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)  
        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if feat == "k":
            feats = k[:, 1:, :]
        elif feat == "q":
            feats = q[:, 1:, :]
        elif feat == "v":
            feats = v[:, 1:, :]
        return feats, normalized_cls_attention
    
    def forward_features(self, imgs):
        try:
            ## for the backbones that does not support the function
            features = self.model.get_intermediate_layers(imgs)[0]
            features = features[:, 1:]
            normalized_cls_attention = self.model.get_last_selfattention(imgs)
        except:
            features = self.model.forward_features(imgs)[:, 1:]
            normalized_cls_attention = None
        return features, normalized_cls_attention

class FeatureExtractorBeta(torch.nn.Module):
    def __init__(self, vit_model, eval_spatial_resolution=14, d_model=768):
        super().__init__()
        self.model = vit_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model
        self.define_forward_features()

    def define_forward_features(self):
        """
        This function defines the forward_features function for the model.
        """
        funcType = MethodType
        device = self.device

        def ff1(self, imgs, feat="k"):
            features, normalized_cls_attention = self.get_intermediate_layer_feats(imgs, feat=feat, layer_num=-1)
            return features, normalized_cls_attention

        def ff2(self, imgs, feat="k"):
            ## for the backbones that does not support the function
            features = self.model.get_intermediate_layers(imgs)[0]
            features = features[:, 1:]
            # Normalizing the attention map
            attentions = self.model.get_last_selfattention(imgs)
            average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
            temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
            normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
            return features, normalized_cls_attention

        def ff3(self, imgs, feat="k"):
            # In case of dinov2 or simimilar models
            features_dict = self.model.forward_features(imgs)
            features = features_dict['x_norm_patchtokens']
            normalized_cls_attention = None
            return features, normalized_cls_attention

        def ff4(self, imgs, feat="k"):
            # Other models
            features = self.model.forward_features(imgs)
            features = features[:, 1:]
            normalized_cls_attention = None
            return features, normalized_cls_attention

        with torch.no_grad():
            (BS, C, H,W) = 32, 3, 224, 224
            ps_imgs = torch.rand((BS, C, H, W)).to(device)

            try:
                ff1(self, ps_imgs)
                # setattr(self, 'forward_features', ff1)
                self.forward_features = funcType(ff1, self)
                return
            except Exception as e:
                pass
            try:
                ff2(self, ps_imgs)
                # setattr(self, 'forward_features', ff2)
                self.forward_features = funcType(ff2, self)
                return
            except:
                pass
            try:
                self.model = self.model.cuda()
                fts, att = ff3(self, ps_imgs.cuda())
                self.forward_features = funcType(ff3, self)
                return
            except Exception as e:
                pass

            # setattr(self, 'forward_features', ff4)
            self.forward_features = funcType(ff4, self)
    
    def freeze_feature_extractor(self, unfreeze_layers=[]):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break

    def get_intermediate_layer_feats(self, imgs, feat="k", layer_num=-1):
        bs, c, h, w = imgs.shape
        imgs = imgs.reshape(bs, c, h, w)
        ## hook to get the intermediate layers
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        self.model._modules["blocks"][layer_num]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        # I think this is not necessary
        # self.model(imgs)
        attentions = self.model.get_last_selfattention(imgs)
        # Scaling factor
        average_cls_attention = torch.mean(attentions[:, :, 0, 1:], dim=1)
        temp_mins, temp_maxs = average_cls_attention.min(dim=1)[0], average_cls_attention.max(dim=1)[0]
        normalized_cls_attention = (average_cls_attention - temp_mins[:, None]) / (temp_maxs[:, None] - temp_mins[:, None])
        # cls_attentions = process_attentions(attentions[:, :, 0, 1:], self.spatial_resolution)  
        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        if feat == "k":
            feats = k[:, 1:, :]
        elif feat == "q":
            feats = q[:, 1:, :]
        elif feat == "v":
            feats = v[:, 1:, :]
        return feats, normalized_cls_attention
    
    @property
    def device(self):
        """
        Returns the device where the model is stored.
        """
        return next(self.model.parameters()).device






        
if __name__ == "__main__":

    img = torch.randn(1, 3, 224, 224)
    dino_vit_s16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino_vit_s16.eval()
    feature_extractor = FeatureExtractor(dino_vit_s16)
    feats, attentions = feature_extractor.get_intermediate_layer_feats(img, feat="k", layer_num=-1)
    print(f"Feats shape : {feats.shape}")
    print(f"Attentions shape : {attentions.shape}")