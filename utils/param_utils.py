from typing import Iterable, Tuple, Union
import torch

PARAMETERS_DTYPE = Union[torch.Tensor, Iterable[torch.Tensor]]

__all__ = ["count_params", "compute_param_norm"]


@torch.no_grad()
def count_params(parameters: PARAMETERS_DTYPE, requires_grad: bool = True) -> Tuple[int, int]:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    count: int = 0
    count_elements: int = 0
    for p in parameters:
        p: torch.Tensor
        if requires_grad and (not p.requires_grad):
            continue
        count += 1
        count_elements += p.numel()
    return count, count_elements


@torch.no_grad()
def compute_param_norm(parameters: PARAMETERS_DTYPE, norm_type: float = 2.0,
                       requires_grad: bool = True) -> torch.Tensor:
    """Compute parameter norm.

    Args:
        parameters:             iterable of parameters (List, Tuple, Iter, ...)
        norm_type (float):      default l2 norm (2.0)
        requires_grad (bool):   whether to count only parameters with requires_grad=True.
    Returns:
        Tensor:              (1,) scalar
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    if requires_grad:
        parameters = [p for p in parameters if p.requires_grad]
    else:
        parameters = list(parameters)
    if len(parameters) == 0:
        return torch.as_tensor(0., dtype=torch.float32)

    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p, norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def param_log(model: torch.nn.Module) -> str:
    s = "-" * 72 + "\n"
    s += "Parameters:\n"
    for p_name, p in model.named_parameters():
        s += f"... {p_name:<60}\t{str(tuple(p.shape)):<20}(std: {torch.std(p, unbiased=False).item():.3f})\n"
    s += "-" * 72 + "\n"
    return s