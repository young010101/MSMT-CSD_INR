from typing import Protocol, Any
from torch import Tensor

from diffusion_calculator import SignalSingleShell, SignalMultishell, VonShell
from datasets import SingleShellDataset, MultiShellDataset

import numpy as np

class OutputCalculator(Protocol):
    def output_from_model_out(self, model_out: Any) -> Any:
        ...


class CoeffDiff:
    def __init__(self, diff_calculator: Any):
        self.diff_calculator = diff_calculator

    def output_from_model_out(self, model_out: Any) -> tuple[Any, dict[str, Any]]:
        diff_signal = self.diff_calculator.compute_signal_from_coeff(model_out)
        neg_signal = self.diff_calculator.compute_negative_signal(model_out)
        return diff_signal, {"neg_values": neg_signal, "model_out": model_out}


def create_singleshell(
    cfg: dict, dataset: SingleShellDataset, device: str, **kwargs
) -> CoeffDiff:
    resp_c = dataset.get_response()
    directions = dataset.get_directions()
    diff_calculator = SignalSingleShell(
        cfg["train_cfg"]["lmax"],
        resp_c,
        cart_bvec=directions,
        device=device,
        fod_rescale=cfg["fod_rescale"],
    )
    return CoeffDiff(diff_calculator)


def create_multishell(
    cfg: dict, dataset: MultiShellDataset, device: str, **kwargs
) -> CoeffDiff:
    """Factory function to create a CoeffDiff instance for multi-shell data.

    Args:
        cfg: Configuration dictionary containing training parameters.
        dataset: An instance of MultiShellDataset providing diffusion data.
        device: Target device for computation (e.g., 'cpu' or 'cuda').
        **kwargs: Additional optional parameters.

    Returns:
        CoeffDiff: A configured instance for computing multi-shell signals.
    """
    bval_idx = [np.where(dataset.shells == bval)[0][0] for bval in dataset.get_bvals()]

    diff_calculator = SignalMultishell(
        cfg["train_cfg"]["lmax"],
        dataset.get_response(),
        bval_idx,
        cart_bvec=dataset.get_directions(),
        device=device,
        fod_rescale=cfg["fod_rescale"],
    )
    return CoeffDiff(diff_calculator)


def create_von(
    cfg: dict, dataset: MultiShellDataset, device: str, **kwargs
) -> CoeffDiff:
    diff_calculator = VonShell()
    return CoeffDiff(diff_calculator)
    

OUTPUT_CALCULATORS = {
    "singleshell": create_singleshell,
    "multishell": create_multishell,
    "von_shell": create_von,
}


def get_output_calculator(cfg: dict, **kwargs) -> OutputCalculator:
    constructor = OUTPUT_CALCULATORS.get(cfg["output_calculator"], None)

    if constructor is None:
        raise Exception("No output calculator for found")
    return constructor(cfg, **kwargs)
