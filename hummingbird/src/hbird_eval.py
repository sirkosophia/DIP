# This code is heavily based on: https://github.com/vpariza/open-hummingbird-eval


from typing import Any, Dict, Tuple, List, Callable, Union, Optional
import os

import numpy as np
import scann
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
from torch.utils.data import DataLoader
from einops import rearrange
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator


from typing import Any, Dict, Tuple, List, Callable, Union, Optional

from hummingbird.src.miou import fast_cm_torch, per_class_iou_torch

from hummingbird.src.models import FeatureExtractorBeta as FeatureExtractor
from hummingbird.src.models import FeatureExtractorSimple

from hummingbird.src.voc_data import VOCDataModule
from hummingbird.src.transforms import get_hbird_val_transforms, get_hbird_train_transforms
from hummingbird.src.image_transformations import CombTransforms
from hummingbird.src.ade20kdata import Ade20kDataModule

from torch.cuda.amp import autocast

def batched_bincount(x: Tensor, max_value: int, dim: int = -1) -> Tensor:
    # adapted from
    # https://discuss.pytorch.org/t/batched-bincount/72819/3
    shape = x.shape[:-1] + (max_value,)
    target = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def histogram_binning(x: Tensor, bins: int) -> Tensor:
    min_val, max_val = x.min(), x.max()
    x = (x - min_val) / (max_val - min_val)
    x = (x * bins).long()
    return x




class HbirdEvaluation():
    def __init__(
        self,
        feature_extractor: FeatureExtractorSimple,
        train_loader: DataLoader,
        num_neighbour: int,
        augmentation_epoch: int,
        num_classes: int,
        device: str,
        evaluation_task: str = 'segmentation',
        num_bins: int = 255,
        is_distributed: bool = False,
        nn_params: Dict[str, Any] = None,
        memory_size: int = None,
        dataset_size: int = None,
        patch_size: int = None,
        f_mem_p: str = None,
        l_mem_p: str = None,
        beta: float = 0.02,
    ) -> None:
        assert evaluation_task in ['segmentation', 'depth'], "Evaluation task should be either segmentation or depth"
        if nn_params is None:
            nn_params = {}
        self.feature_extractor = feature_extractor
        self.device = device
        self.is_distributed = is_distributed
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.num_neighbour = num_neighbour
        self.num_bins = num_bins
        self.evaluation_task = evaluation_task
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_extractor = self.feature_extractor.half()
        self.num_classes = num_classes
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.num_sampled_features = None
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p
        self.beta = beta 

        if self.evaluation_task == 'segmentation':
            self.gt_key = 'mask'
        elif self.evaluation_task == 'depth':
            self.gt_key = 'depth'

        if not self.load_memory():
            if self.memory_size is not None:
                dataset_size = len(train_loader) * (dataset_size // len(train_loader))
                            
                self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch)
                self.num_sampled_features = min(self.num_sampled_features, eval_spatial_resolution**2)
                self.memory_size = dataset_size * self.augmentation_epoch * self.num_sampled_features
                print("Number of sampled features: ", self.num_sampled_features)
                print("effective memory size: ", self.memory_size)
                ## create memory of specific size
                self.feature_memory = torch.zeros((self.memory_size, self.feature_extractor.d_model))
                if self.evaluation_task == 'segmentation':
                    self.label_memory = torch.zeros((self.memory_size, self.num_classes))
                elif self.evaluation_task == 'depth':
                    self.label_memory = torch.zeros((self.memory_size, patch_size * patch_size))
            self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution)
            self.save_memory()
        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        # breakpoint()
        norm = torch.norm(self.feature_memory, dim=1)
        # check if some zeros or nans
        if torch.any(norm == 0) or torch.any(torch.isnan(norm)):
            raise ValueError("Some features have norm 0 or nan")

    def create_NN(
        self,
        num_neighbour: int = 30,
        num_leaves_to_search: Optional[int] = None,
        num_leaves: Optional[int] = None,
        num_reordering_candidates: Optional[int] = None,
        distance_measure: str = "dot_product",
        anisotropic_quantization_threshold: float = 0.2,
    ) -> None:
        """
        following advices from:
        https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
        """
        if num_reordering_candidates is None:
            # num_reordering_candidates = num_neighbour * 10
            num_reordering_candidates = 120
            print("Number of reordering candidates is not provided, setting it to ", num_reordering_candidates)
        if num_leaves is None:
            num_leaves = 512
            print("Number of leaves is not provided, setting it to ", num_leaves)
        if num_leaves_to_search is None:
            num_leaves_to_search = 32
            print("Number of leaves to search is not provided, setting it to ", num_leaves_to_search)
        print("feature memory size: ", self.feature_memory.size())
        self.NN_algorithm = scann.scann_ops_pybind.builder(
            self.feature_memory.detach().cpu().numpy(),
            num_neighbour,
            distance_measure,
        )
        print("after builder")
        self.NN_algorithm = self.NN_algorithm.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=self.feature_memory.size(0),
        )
        print("after tree")
        self.NN_algorithm = self.NN_algorithm.score_ah(
            2,
            anisotropic_quantization_threshold=anisotropic_quantization_threshold
        )
        print("after score")
        self.NN_algorithm = self.NN_algorithm.reorder(
            num_reordering_candidates,
        )
        print("after reorder")
        # breakpoint()
        self.NN_algorithm = self.NN_algorithm.build()
        print("after build")
    
    @torch.no_grad()
    def find_nn_simple(self, q):
        flat_input = q #.view(-1, self.feature_extractor.d_model)
        feature_memory_chunk_size = 102400  # Chunk size for feature_memory
        N, M = flat_input.size(0), self.feature_memory.size(0)  # Sizes
        top_k = self.num_neighbour  # Number of closest neighbors to find

        # Initialize top-k storage for all rows of flat_input
        top_distances = torch.full((N, top_k), float('inf'), device=flat_input.device)  # Shape: [N, top_k]
        top_indices = torch.full((N, top_k), -1, device=flat_input.device, dtype=torch.long)  # Shape: [N, top_k]

        row_squared_sum = torch.sum(flat_input**2, dim=1, keepdim=True)  # Precompute squared sums for all rows of flat_input

        for mem_start in range(0, M, feature_memory_chunk_size):  # Iterate over feature_memory in chunks
            feature_memory_chunk = self.feature_memory[mem_start:mem_start + feature_memory_chunk_size]  # Shape: [chunk_size, D]
            feature_memory_squared_sum_chunk = torch.sum(feature_memory_chunk**2, dim=1)  # Shape: [chunk_size]

            # Compute distances between all flat_input and current feature_memory chunk
            dot_product_chunk = torch.matmul(flat_input, feature_memory_chunk.t()) 
            distances_chunk = row_squared_sum + feature_memory_squared_sum_chunk - 2 * dot_product_chunk  # Shape: [N, chunk_size]

            # Combine current top-k distances with distances from the current chunk
            combined_distances = torch.cat((top_distances, distances_chunk), dim=1)  # Shape: [N, top_k + chunk_size]
            combined_indices = torch.cat((
                top_indices,
                torch.arange(mem_start, mem_start + feature_memory_chunk.size(0), device=flat_input.device).repeat(N, 1)
            ), dim=1)  # Shape: [N, top_k + chunk_size]

            # Select top-k distances and corresponding indices
            new_top_distances, topk_indices = torch.topk(-combined_distances, k=top_k, dim=1, largest=True)
            new_top_indices = torch.gather(combined_indices, 1, topk_indices)

            # Update the global top-k distances and indices
            top_distances = -new_top_distances  # Revert the negative sign
            top_indices = new_top_indices
        return top_indices

    @torch.no_grad()
    def create_memory(self, train_loader: DataLoader, num_classes: int, eval_spatial_resolution: int) -> None:
        feature_memory = list()
        label_memory = list()
        idx = 0
        for _ in range(self.augmentation_epoch):
            for _, (x, y) in enumerate(tqdm(train_loader, desc='Memory Creation loop')):
                x = x.to(self.device, dtype=torch.float16)
                y = y.to(self.device)
                # import ipdb; ipdb.set_trace(context=21)
                with autocast():
                    features, _ = self.feature_extractor.forward_features(x)
                patch_size = x.shape[-1] // eval_spatial_resolution
                if self.evaluation_task == 'segmentation':
                    # y = y.long()
                    y = (y * 255).long()
                    y[y == 255] = 0
                    patchified_gts = self.patchify_gt(y, patch_size)  # (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)
                elif self.evaluation_task == 'depth':
                    patchified_gts = self.patchify_depth(y, patch_size)
                    label = patchified_gts

                if self.memory_size is None:
                    # Memory Size is unbounded so we store all the features
                    normalized_features = features #

                    normalized_features = normalized_features.flatten(0, 1)
                    label = label.flatten(0, 2)
                    feature_memory.append(normalized_features.detach().cpu())
                    label_memory.append(label.detach().cpu())
                else:
                    # Memory Size is bounded so we need to select/sample some features only
                    sampled_features, sampled_indices = self.sample_features_batch(features, patchified_gts)
                    normalized_sampled_features = sampled_features #
                    label = label.flatten(1, 2)
                    ## select the labels of the sampled features
                    sampled_indices = sampled_indices.to(self.device)
                    ## repeat the label for each sampled feature
                    label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                    normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                    label_hat = label_hat.flatten(0, 1)
                    # breakpoint()
                    self.feature_memory[idx:idx + normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                    self.label_memory[idx:idx + label_hat.size(0)] = label_hat.detach().cpu()
                    idx += normalized_sampled_features.size(0)

        if self.memory_size is None:
            self.feature_memory = torch.cat(feature_memory)
            self.label_memory = torch.cat(label_memory)
            if self.is_distributed:
                raise NotImplementedError("Distributed training is not supported for unbounded memory size")

        elif self.is_distributed:
            dist.barrier()
            self.feature_memory = self.feature_memory.to(self.device, dtype=torch.float16)
            self.label_memory = self.label_memory.to(self.device, dtype=torch.float16)
            receive_features = [torch.zeros_like(self.feature_memory) for _ in range(dist.get_world_size())]
            receive_labels = [torch.zeros_like(self.label_memory) for _ in range(dist.get_world_size())]
            dist.all_gather(receive_features, self.feature_memory)
            dist.all_gather(receive_labels, self.feature_memory)
            self.feature_memory = torch.cat(receive_features)
            self.label_memory = torch.cat(receive_labels)
        self.feature_memory = self.feature_memory.to(dtype=torch.float16)
        self.label_memory = self.label_memory.to(dtype=torch.float16)

    def save_memory(self) -> None:
        if self.is_distributed:
            if dist.get_rank() != 0:
                return
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)

    def load_memory(self) -> bool:
        if self.f_mem_p is None:
            return False
        if os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p).to(self.device, dtype=torch.float16)
            self.label_memory = torch.load(self.l_mem_p).to(self.device, dtype=torch.float16)
            return True
        return False

    def sample_features_batch(self, features: Tensor, pathified_gts: Tensor) -> Tuple[Tensor, Tensor]:
        the_max_value = self.num_classes
        if self.evaluation_task == 'depth':
            # Here bin the continuous depth values into discrete bins
            # That are subsequently used to sample the features similarly
            # to the classes in the segmentation task
            pathified_gts = histogram_binning(pathified_gts, bins=self.num_bins)
            the_max_value = self.num_bins + 1

        batch_size = features.size(0)

        unique_classes_per_patch = batched_bincount(pathified_gts, max_value=the_max_value, dim=-1) > 0
        class_frequency = unique_classes_per_patch.sum(dim=(1, 2))
        patch_scores = (class_frequency.view(batch_size, 1, 1, -1) * unique_classes_per_patch).sum(-1).float()
        nonzero_indices = unique_classes_per_patch.sum(dim=-1) > 0

        patch_scores = patch_scores.flatten(start_dim=1)
        nonzero_indices = nonzero_indices.flatten(start_dim=1)
        patch_scores[~nonzero_indices] = 1e6

        uniform_x = torch.rand_like(patch_scores[nonzero_indices])
        patch_scores[nonzero_indices] *= uniform_x
        # breakpoint()
        _, sampled_indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)

        sampled_features = features.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, features.shape[-1]))

        return sampled_features, sampled_indices

    def patchify_gt(self, gt: Tensor, patch_size: int) -> Tensor:
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)
        return gt

    def patchify_depth(self, gt: Tensor, patch_size: int) -> Tensor:
        gt = gt[:, 0:1]  # we are storing 'gray scale' depth values
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)
        return gt

    def cross_attention(self, q: Tensor, k: Tensor, v: Tensor, beta: float = 0.02) -> Tensor:
        """
        Args:
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)
        """
        # d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2)  # (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat

    def find_nearest_key_to_query(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs * num_patches, d_k)

        neighbors = self.find_nn_simple(torch.from_numpy(reshaped_q).to(self.device))
        
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.num_neighbour, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.num_neighbour, -1)
        return key_features, key_labels


    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, eval_spatial_resolution: int, return_labels: bool = False) -> Tuple[Dict[str, Union[float, List[float]]], Tensor]:  # type: ignore
        self.feature_extractor = self.feature_extractor.to(self.device)

        if self.evaluation_task == 'segmentation':
            # ...accumulated confusion matrices for class labels, valid pixels, and invalid pixels (in pixel counts)
            running_metric = torch.zeros((1, self.num_classes, self.num_classes), device=self.device)
        elif self.evaluation_task == 'depth':
            running_metric = DepthMetrics(device=self.device)
        print("in evaluate function")
        num_frames = 1
        label_hats = None
        start_idx = 0
        
        for i, (x, y) in enumerate(tqdm(val_loader, desc='Evaluation loop')):

            x = x.to(self.device)
            *_, h, w = x.shape
            features, _ = self.feature_extractor.forward_features(x.to(self.device, dtype=torch.float16))
            features = features.to(self.device)
            y = y.to(self.device)
            y = (y * 255).long()
            ## copy the data of features to another variable
            q = features.clone()
            q = q.detach().cpu().numpy()
            key_features, key_labels = self.find_nearest_key_to_query(q)
            label_hat = self.cross_attention(features, key_features, key_labels, beta=self.beta)
            bs, _, label_dim = label_hat.shape
            label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
            resized_label_hats = F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
            if self.evaluation_task == 'segmentation':
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)

            if return_labels:
                if (label_hats is None):
                    if self.evaluation_task == 'segmentation':
                        size = (len(val_loader.dataset), *cluster_map.shape[1:])
                    elif self.evaluation_task == 'depth':
                        size = (len(val_loader.dataset), *resized_label_hats.shape[1:])
                    label_hats = torch.empty(size, dtype=label_hat.dtype, device='cpu')

                if self.evaluation_task == 'segmentation':
                    label_hats[start_idx:start_idx + cluster_map.size(0) ] = rearrange(cluster_map.cpu(), '(b t) ... -> b t ...', t=num_frames)
                    start_idx += cluster_map.size(0)
                elif self.evaluation_task == 'depth':
                    label_hats[start_idx:start_idx + resized_label_hats.size(0) ] = rearrange(resized_label_hats.cpu(), '(b t) ... -> b t ...', t=num_frames)
                    start_idx += resized_label_hats.size(0) 

            if self.evaluation_task == 'segmentation':
                valid_idx = y != 255
                running_metric += fast_cm_torch(y[valid_idx].unsqueeze(0), cluster_map[valid_idx].unsqueeze(0), self.num_classes, do_check=False)
            elif self.evaluation_task == 'depth':
                running_metric.update(y, resized_label_hats)

        if self.is_distributed:
            dist.barrier()
            if self.evaluation_task == 'segmentation':
                dist.all_reduce(running_metric, op=dist.ReduceOp.SUM)
            elif self.evaluation_task == 'depth':
                running_metric.dist_all_reduce()

        if self.evaluation_task == 'segmentation':
            # Metrics based on the ground-truth class labels <= cm, cm_valid, cm_invalid
            # ...mean intersection over union (mIoU) for all pixels, valid pixels, and invalid pixels
            iou_valid = per_class_iou_torch(running_metric).view(-1)
            miou_valid = torch.nanmean(iou_valid)
            # breakpoint()
            logs = {'mIoU': miou_valid.item(), 'IoU': iou_valid.tolist()}
        elif self.evaluation_task == 'depth':
            logs = running_metric.compute()

        return logs, label_hats


def hbird_evaluation(
    model: nn.Module,
    d_model: int,
    data_dir:str,
    ftr_extr_fn: Callable[[nn.Module, Tensor, bool], Tensor],
    patch_size: int,
    dataset_name: str,
    dataset_args: Dict[str, Any] = {},
    evaluation_task: str = 'segmentation',
    num_bins: int = 255,
    batch_size: int = 64,
    batch_size_eval: int = 64,
    augmentation_epoch: int = 1,
    device: str = 'gpu',
    is_distributed: bool = False,
    num_workers: int = 8,
    return_labels: bool = False,
    num_neighbour: int = 30,
    nn_params: Dict[str, Any] = None,
    memory_size: int = None,
    f_mem_p: str = None,
    l_mem_p: str = None,
    input_size: int = 224,
    beta: float = 0.02,
) -> Tuple[Dict[str, Union[float, List[float]]], Tensor]:
    
    IMAGNET_MEAN = [0.485, 0.456, 0.406]
    IMAGNET_STD = [0.229, 0.224, 0.255]

    img_mean = IMAGNET_MEAN
    img_std = IMAGNET_STD
    print("Using ImageNet mean and std")
        
    eval_spatial_resolution = input_size // patch_size
    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    else:
        feature_extractor = FeatureExtractorSimple(model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    

    train_transforms_dict = get_hbird_train_transforms(input_size, img_mean=img_mean, img_std=img_std, is_radio=False)
    val_transforms_dict = get_hbird_val_transforms(input_size, img_mean=img_mean, img_std=img_std)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])

    dataset_size = 0
    num_classes = 0
   
    if dataset_name == "voc":
        dataset = VOCDataModule(batch_size=batch_size,
                                    num_workers=num_workers,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True)
        dataset.setup()
    elif dataset_name == "ade20k":
        dataset = Ade20kDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size)
        dataset.setup()
    else:
        raise ValueError("Unknown dataset name")
    
    dataset_size = dataset.get_train_dataset_size()
    num_classes = dataset.get_num_classes()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    print("Dataset size: ", dataset_size)
    print("Number of classes: ", num_classes)
    
    evaluator = HbirdEvaluation(
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        num_neighbour=num_neighbour,
        augmentation_epoch=augmentation_epoch,
        num_classes=num_classes,
        evaluation_task=evaluation_task,
        num_bins=num_bins,
        device=device,
        is_distributed=is_distributed,
        nn_params=nn_params,
        memory_size=memory_size,
        dataset_size=dataset_size,
        patch_size=patch_size,
        f_mem_p=f_mem_p,
        l_mem_p=l_mem_p,
        beta=beta,
    )
    print("before eval")
    return evaluator.evaluate(val_loader, eval_spatial_resolution, return_labels=return_labels)


