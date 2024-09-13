import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence


BAR_FORMAT = "{l_bar}{bar:15}{r_bar}"

def float_separator(num: int) -> str:
    num_seg = []
    while num > 1000:
        num_seg.append(num % 1000)
        num = num // 1000

    str_num = [num] + num_seg[::-1]
    temp = []
    for i, n in enumerate(str_num):
        if n == 0:
            temp.append('000')
        elif (i != 0) and (n < 100):
            temp.append('0' + str(n))
        else:
            temp.append(str(n))
    str_num = ','.join(temp)
    return str_num


def init_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # one gpu
    torch.cuda.manual_seed_all(seed)    # multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n* Set a random seed to {seed}\n")


class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adagrad
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adagrad

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def trim_pad(input_ids, input_embeddings, pad_value=0, side="right"):
    compressed = torch.sum(input_ids, dim=0)  # (N, D) or (N, )

    drop_map = (compressed == pad_value).to(torch.long).to(input_ids.device)
    if torch.sum(drop_map) == 0:
        return input_ids, input_embeddings

    if side == 'right':
        drop_start_idx = torch.argmax(drop_map * torch.arange(len(drop_map), 0, -1).to(input_ids.device))
        return input_ids[:, :drop_start_idx], input_embeddings[:, :drop_start_idx]
    elif side == 'left':
        drop_start_idx = torch.argmax(drop_map * torch.arange(len(drop_map)).to(input_ids.device))
        return input_ids[:, drop_start_idx + 1:], input_embeddings[:, drop_start_idx + 1:]
    else:
        raise NotImplementedError


def repeat_interleave(input_embeddings, repeats, dim=None, batch_first=True, padding_value=0):
    """
    The 'inputs' and 'repeats' in this method must have the original form.
    It means that the shape of the 'inputs' and 'repeats' should not be reshaped.
    Then, in this function, the dim of the 'inputs' must be less than 2 and the dim of the 'repeats' should be  1.
    """

    # Calculate the shape of the output array and Determine the number of dimensions in the input
    output_shape = list(input_embeddings.shape)
    device = input_embeddings.device

    repeat_num = []
    if (type(repeats) == int) or (repeats.ndim == 1 and len(repeats) == 1):
        """
        In the case of the "candidates" and "block_scores" in getting latent subword representation.
        """
        # get the shape of the output
        output_shape[dim] *= repeats
        # reshape the input_embeddings
        input_embeddings = input_embeddings.flatten(start_dim=0, end_dim=1)
    elif repeats.ndim == 2:
        if (input_embeddings.ndim - repeats.ndim) == 1:  # 3dim(not flattened) - 2dim(not flattened)
            """
            In the case of the "input_embeddings"
            """
            output_shape[dim] = int(torch.sum(repeats) / output_shape[0])
            # for padding
            repeat_num = torch.sum(repeats, dim=dim, keepdim=False)
            max_repeat_num = torch.max(repeat_num)
            # get the shape of the output
            output_shape[dim] = int(max_repeat_num)
            # reshape the inputs and repeats
            input_embeddings = input_embeddings.flatten(start_dim=0, end_dim=1)  # inputs.ndim = 2   (BxN, D)
            repeats = repeats.flatten()  # repeats.ndim = 1  (BxN, )
            assert input_embeddings.shape[0] == repeats.shape[0]
        elif (input_embeddings.ndim - repeats.ndim) == 0:  # 2dim(not flattened) - 2dim(not flattened)
            """
            In the case of the "context_input_ids"
            """
            # for padding
            repeat_num = torch.sum(repeats, dim=dim, keepdim=False)
            max_repeat_num = torch.max(repeat_num)
            # get the shape of the output
            output_shape[dim] = int(max_repeat_num)
            # reshape the input_embeddings and repeats
            input_embeddings = input_embeddings.unsqueeze(-1)
            input_embeddings = input_embeddings.flatten(start_dim=0, end_dim=1)  # input_embeddings.ndim = 2   (BxN, 1)
            repeats = repeats.flatten()  # repeats.ndim = 1  (BxN, )
            assert input_embeddings.shape[0] == repeats.shape[0]
        else:
            raise NotImplementedError
    elif repeats.ndim == 1:
        if (input_embeddings.ndim - repeats.ndim) == 2:  # 3dim(not flattened) - 1dim(not flattened)
            output_shape[dim] = int(torch.sum(repeats))
            # reshape the input_embeddings and repeats
            input_embeddings = input_embeddings.flatten(start_dim=0, end_dim=1)  # input_embeddings.ndim = 2   (BxN, D)
            repeats = repeats.repeat(len(input_embeddings) // len(repeats))
            assert input_embeddings.shape[0] == repeats.shape[0]
        elif (input_embeddings.ndim - repeats.ndim) == 1:  # 2dim(not flattened) - 1dim(not flattened)
            output_shape[dim] = int(torch.sum(repeats))
            # reshape the input_embeddings and repeats
            input_embeddings = input_embeddings.unsqueeze(-1)
            input_embeddings = input_embeddings.flatten(start_dim=0, end_dim=1)  # input_embeddings.ndim = 2   (BxN, D)
            repeats = repeats.repeat(len(input_embeddings) // len(repeats))
            assert input_embeddings.shape[0] == repeats.shape[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Create an array of repeated indices along the specified dim
    indices = np.repeat(np.arange(input_embeddings.shape[0]), repeats=repeats, axis=0)
    indices = torch.LongTensor(indices).to(device)

    # take to select the elements based on the repeated indices
    output = input_embeddings[indices]

    # Reshape the output array to match the desired shape
    try:
        # print("normal_reshape")
        output = output.reshape(output_shape)
        return output
    except RuntimeError:
        # print("padding_reshape")
        padded_output = []
        cur_idx = 0
        for repeat in repeat_num:
            repeat = int(repeat)
            padded_output.append(output[cur_idx: cur_idx + repeat])
            cur_idx += repeat
        padded_output = pad_sequence(padded_output, batch_first=batch_first, padding_value=padding_value)
        return padded_output.squeeze()