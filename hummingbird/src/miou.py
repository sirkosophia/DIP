"""
Copy-pasted from the following source:
https://github.com/valeoai/bravo_challenge/blob/main/bravo_toolkit/eval/metrics.py
"""
from typing import Optional

import numpy as np
import torch


def batched_bincount(x: torch.Tensor, max_value: int, dim: int = -1) -> torch.Tensor:
    # adapted from
    # https://discuss.pytorch.org/t/batched-bincount/72819/3
    shape = (len(x), max_value)
    target = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def fast_cm_torch(y_true: torch.Tensor, y_pred: torch.Tensor, n: int, do_check: bool = True, invalid_value: Optional[int] = None) -> torch.Tensor:
    '''
    Fast computation of a confusion matrix from two arrays of labels.

    Args:
        y_true  (torch.Tensor): array of true labels
        y_pred (torch.Tensor): array of predicted labels
        n (int): number of classes

    Returns:
        torch.Tensor: confusion matrix, where rows are true labels and columns are predicted labels
    '''
    y_true = y_true.flatten(start_dim=1).long()
    y_pred = y_pred.flatten(start_dim=1).long()

    if do_check:
        k = (y_true < 0) | (y_true > n) | (y_pred < 0) | (y_pred > n)
        if torch.any(k):
            raise ValueError(f'Invalid class values in ground-truth or prediction: {torch.unique(torch.cat((y_true[k], y_pred[k])))}')

    # Convert class numbers into indices of a simulated 2D array of shape (n, n) flattened into 1D, row-major
    effective_indices = n * y_true + y_pred
    max_value = n ** 2
    if invalid_value is not None:
        max_value = n ** 2 + 1
        effective_indices[y_true == invalid_value] = n ** 2
    # Count the occurrences of each index, reshaping the 1D array into a 2D array
    return batched_bincount(effective_indices, max_value)[..., :n ** 2].view(-1, n, n)


def per_class_iou_torch(cm: torch.Tensor) -> torch.Tensor:
    ''''
    Compute the Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        cm (torch.Tensor): n x n 2D confusion matrix (the orientation is not important, as the formula is symmetric)

    Returns:
        torch.Tensor: 1D array of IoU values for each of the n classes
    '''
    # The diagonal contains the intersection of predicted and true labels
    # The sum of rows (columns) is the union of predicted (true) labels (or vice-versa, depending on the orientation)
    return torch.diagonal(cm, dim1=1, dim2=2) / (cm.sum(2) + cm.sum(1) - torch.diagonal(cm, dim1=1, dim2=2))


def fast_cm(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    '''
    Fast computation of a confusion matrix from two arrays of labels.

    Args:
        y_true  (np.ndarray): array of true labels
        y_pred (np.ndarray): array of predicted labels
        n (int): number of classes

    Returns:
        np.ndarray: confusion matrix, where rows are true labels and columns are predicted labels
    '''
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    k = (y_true < 0) | (y_true > n) | (y_pred < 0) | (y_pred > n)
    if np.any(k):
        raise ValueError('Invalid class values in ground-truth or prediction: '
                         f'{np.unique(np.concatenate((y_true[k], y_pred[k])))}')
    # Convert class numbers into indices of a simulated 2D array of shape (n, n) flattened into 1D, row-major
    effective_indices = n * y_true + y_pred
    # Count the occurrences of each index, reshaping the 1D array into a 2D array
    return np.bincount(effective_indices, minlength=n ** 2).reshape(n, n)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    ''''
    Compute the Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        cm (np.ndarray): n x n 2D confusion matrix (the orientation is not important, as the formula is symmetric)

    Returns:
        np.ndarray: 1D array of IoU values for each of the n classes
    '''
    # The diagonal contains the intersection of predicted and true labels
    # The sum of rows (columns) is the union of predicted (true) labels (or vice-versa, depending on the orientation)
    return np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))


if __name__ == '__main__':
    # Example usage
    y_true = np.array([[0, 0, 1], [1, 0, 2]])
    y_pred = np.array([[0, 1, 1], [1, 2, 2]])
    # print(y_true.shape)
    n = 3
    cm = fast_cm(y_true, y_pred, n)
    iou = per_class_iou(cm)
    miou = np.nanmean(iou)
    print(miou)

    # test with another example
    y_true_2 = np.array([[2, 0, 1], [0, 2, 2]])
    y_pred_2 = np.array([[0, 1, 1], [0, 0, 2]])
    cm_2 = fast_cm(y_true_2, y_pred_2, n)
    iou_2 = per_class_iou(cm_2)
    miou_2 = np.nanmean(iou_2)
    print(miou_2)

    # test with torch and concatenate the two examples
    y_true_torch = torch.from_numpy(np.stack([y_true, y_true_2], axis=0))
    y_pred_torch = torch.from_numpy(np.stack([y_pred, y_pred_2], axis=0))
    # print(y_true_torch.shape)
    cm_torch = fast_cm_torch(y_true_torch, y_pred_torch, n)
    iou_torch = per_class_iou_torch(cm_torch)
    miou_torch = torch.nanmean(iou_torch, dim=-1)
    print(miou_torch)
