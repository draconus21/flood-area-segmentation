import numpy as np
from torch import Tensor


def cuda_to_numpy(tensor: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif not isinstance(tensor, Tensor):
        raise ValueError(f"Must be of type np.ndarray to torch.Tensor, got {type(tensor)}")

    out: Tensor = tensor.detach() if tensor.requires_grad else tensor
    out = out.cpu() if out.device.type == "cuda" else out

    return out.numpy()


def tensor_to_numpy(tensor: np.ndarray | Tensor, keep_batch: bool = False) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif not isinstance(tensor, Tensor):
        raise ValueError(f"Must be of type np.ndarray to torch.Tensor, got {type(tensor)}")

    # if not hasattr(tensor, 'squeeze'):
    #    return cuda_to_numpy(tensor)

    out: Tensor = tensor if keep_batch else tensor.squeeze()
    return cuda_to_numpy(out)
