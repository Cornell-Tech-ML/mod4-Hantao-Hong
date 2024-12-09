from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    h = height // kh
    w = width // kw

    out = input.contiguous().view(batch, channel, height, w, kw)
    out = out.permute(0, 1, 3, 2, 4)
    out = out.contiguous().view(batch, channel, h, w, kh * kw)

    return out, h, w


# TODO: Implement for Task 4.3.
def avgpool2d(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to a 2D input tensor.

    Args:
    ----
        input_tensor: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kh, kw) representing the height and width of the pooling kernel

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width) after applying average pooling

    """
    batch, channel, _, _ = input_tensor.shape

    input_tensor, h, w = tile(input_tensor, kernel)
    out = input_tensor.mean(dim=4)

    return out.view(batch, channel, h, w)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input_tensor: Tensor to compute argmax on.
        dim: Dimension to reduce.

    Returns:
    -------
        Tensor with the argmax as a 1-hot tensor.

    """
    out = max_reduce(input_tensor, dim)
    return out == input_tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Max function.

        Args:
        ----
            ctx: Context object to save information for backward computation.
            input_tensor: Input tensor to apply the max function on.
            dim: Dimension to reduce.

        Returns:
        -------
            Tensor after applying the max function along the specified dimension.

        """
        ctx.save_for_backward(input_tensor, dim)
        return max_reduce(input_tensor, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Max function.

        Args:
        ----
            ctx: Context object with saved information from the forward pass.
            grad_output: Gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple containing the gradient of the loss with respect to the input tensor and the dimension.
        
        """
        input_tensor, dim = ctx.saved_values
        return (argmax(input_tensor, int(dim.item())) * grad_output, dim)


def max(input_tensor: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input_tensor: Tensor to apply the max function on.
        dim: Dimension to reduce.

    Returns:
    -------
        Tensor after applying the max function along the specified dimension.

    """
    return Max.apply(input_tensor, input_tensor._ensure_tensor(dim))


def softmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the softmax of a tensor along a specified dimension.

    Args:
    ----
        input_tensor: Tensor to compute the softmax on.
        dim: Dimension to reduce.

    Returns:
    -------
        Tensor after applying the softmax function along the specified dimension.

    """
    x = input_tensor.exp()
    return x / x.sum(dim=dim)


def logsoftmax(input_tensor: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor along a specified dimension.

    Args:
    ----
        input_tensor: Tensor to compute the logsoftmax on.
        dim: Dimension to reduce.

    Returns:
    -------
        Tensor after applying the logsoftmax function along the specified dimension.

    """
    x_i = max(input_tensor, dim)
    return input_tensor - ((input_tensor - x_i).exp().sum(dim=dim).log() + x_i)


def maxpool2d(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to a 2D input tensor.

    Args:
    ----
        input_tensor: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kh, kw) representing the height and width of the pooling kernel

    Returns:
    -------
        Tensor of shape (batch, channel, new_height, new_width) after applying max pooling

    """
    batch, channel, _, _ = input_tensor.shape
    input_tensor, h, w = tile(input_tensor, kernel)
    out = max(input_tensor, 4)

    return out.view(batch, channel, h, w)


def dropout(input_tensor: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input_tensor: Tensor to apply dropout on.
        rate: Dropout rate, the probability of setting a unit to zero.
        ignore: If True, dropout is not applied.

    Returns:
    -------
        Tensor after applying dropout.

    """
    if ignore:
        return input_tensor
    else:
        return input_tensor * (rand(input_tensor.shape) > rate)
