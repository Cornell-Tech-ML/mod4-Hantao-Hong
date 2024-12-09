from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            vals (ScalarLike): The input values to the function.

        Returns:
        -------
            Scalar: The result of applying the function to the input values.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the addition function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of adding the two input values.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Performs the backward pass of the addition function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the loss with respect to the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the log function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the log function to the input value.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the log function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the inverse function to the input value.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of multiplying the two input values.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Performs the backward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the loss with respect to the inputs.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the negation function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the negation function to the input value.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the negation function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the sigmoid function to the input value.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the ReLU function to the input value.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the less than function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of comparing the two input values.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Performs the backward pass of the less than function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the loss with respect to the inputs.

        """
        return 0.0, 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the exponential function to the input value.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient of the loss with respect to the input.

        """
        out: float = ctx.saved_values[0]
        return out * d_output


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass of the equality function.

        Args:
        ----
            ctx (Context): The context to save information for backward computation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of comparing the two input values.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Performs the backward pass of the equality function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the loss with respect to the inputs.

        """
        return 0.0, 0.0
