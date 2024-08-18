import torch
import numpy as np


def cuda_to_numpy(tensor: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Must be of type np.ndarray to torch.Tensor, got {type(tensor)}")

    out: torch.Tensor = tensor.detach() if tensor.requires_grad else tensor
    out = out.cpu() if out.device.type == "cuda" else out

    return out.numpy()


def tensor_to_numpy(
    tensor: np.ndarray | torch.Tensor, keep_batch: bool = False, channels_last: bool = False
) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Must be of type np.ndarray to torch.Tensor, got {type(tensor)}")

    # if not hasattr(tensor, 'squeeze'):
    #    return cuda_to_numpy(tensor)

    out: torch.Tensor = tensor
    if channels_last:
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        if len(out.shape) == 3:
            out = out.unsqueeze(0)

        out = out.permute(0, 2, 3, 1)

    if not keep_batch:
        out = out.squeeze()

    return cuda_to_numpy(out)
