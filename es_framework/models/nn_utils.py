import numpy as np
import torch
from typing import OrderedDict as OrderedDictType, Dict


def flatten_nn_parameters(model: torch.nn.Module) -> np.ndarray:
    """
    Flattens all parameters of a PyTorch model into a single NumPy array.
    """
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])


def unflatten_nn_parameters(flat_params: np.ndarray, model_ref: torch.nn.Module) -> OrderedDictType[str, torch.Tensor]:
    """
    Converts a flat NumPy array of parameters back into a PyTorch state_dict.
    Args:
        flat_params (np.ndarray): Flat array of parameters.
        model_ref (torch.nn.Module): A reference model instance to get parameter shapes and names.
    Returns:
        OrderedDict[str, torch.Tensor]: The state dictionary.
    """
    new_state_dict = OrderedDictType()
    current_idx = 0
    for name, param_ref in model_ref.named_parameters():
        num_elements = param_ref.numel()
        shape = param_ref.shape

        # Extract the slice for the current parameter
        param_slice = flat_params[current_idx:current_idx + num_elements]

        # Reshape and convert to tensor
        new_state_dict[name] = torch.from_numpy(param_slice).reshape(shape).float()
        current_idx += num_elements

    if current_idx != flat_params.size:
        raise ValueError(f"Size mismatch: flat_params has {flat_params.size} elements, "
                         f"but model requires {current_idx}.")
    return new_state_dict


def load_flat_params_into_model(flat_params: np.ndarray, model: torch.nn.Module):
    """
    Convenience function to directly load flat parameters into a model.
    """
    state_dict = unflatten_nn_parameters(flat_params, model)
    model.load_state_dict(state_dict)
