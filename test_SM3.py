import torch
import numpy
import pytest
from SM3 import SM3


def _test_dense_0d_updates(lr, momentum, beta):
    var = numpy.array(0.5)
    grad = numpy.array(0.1)

    x = _create_tensor(var, grad)

    opt = SM3([x], lr=lr, momentum=momentum, beta=beta)
    gbar = numpy.zeros_like(grad)
    accumulator = numpy.zeros_like(grad)
    for _ in range(5):
        # Run a step of training.
        opt.step()

        if beta > 0.:
            accumulator = beta * accumulator + (1. - beta) * numpy.square(grad)
        else:
            accumulator = accumulator + numpy.square(grad)

        exp_p_grad = grad / numpy.sqrt(accumulator)
        gbar = momentum * gbar + (1. - momentum) * exp_p_grad
        var = var - lr * gbar
        # Check that variable and momentum are as expected after one step of
        # training.
        _check_values(x.numpy(), var)
        if momentum > 0.:
            _check_values(opt.state[x]['momentum_buffer'].numpy(), gbar)


def _test_dense_1d_updates(lr, momentum, beta):
    var = numpy.array([0.5, 0.3])
    grad = numpy.array([0.1, 0.1])

    x = _create_tensor(var, grad)

    opt = SM3([x], lr=lr, momentum=momentum, beta=beta)

    gbar = numpy.zeros_like(grad)
    accumulator = numpy.zeros_like(grad)
    for _ in range(5):
        # Run a step of training.
        opt.step()

        if beta > 0.:
            accumulator = beta * accumulator + (1. - beta) * numpy.square(grad)
        else:
            accumulator = accumulator + numpy.square(grad)

        exp_p_grad = grad / numpy.sqrt(accumulator)
        gbar = momentum * gbar + (1. - momentum) * exp_p_grad
        var = var - lr * gbar
        # Check that variable and momentum are as expected after one step of
        # training.
        _check_values(x.numpy(), var)
        if momentum > 0.:
            _check_values(opt.state[x]['momentum_buffer'].numpy(), gbar)

def _test_dense_2d_updates(lr, momentum, beta):
    var = numpy.array([[.5, .5], [.5, .5]])
    grad = numpy.array([[.1, .05], [.02, .03]])

    x = _create_tensor(var, grad)

    opt = SM3([x], lr=lr, momentum=momentum, beta=beta)

    row_accumulator = numpy.zeros([2, 1])
    col_accumulator = numpy.zeros([1, 2])
    gbar = numpy.zeros_like(grad)
    accumulator = numpy.zeros_like(grad)
    for _ in range(5):
        # Run a step of training.
        opt.step()

        accumulator = numpy.minimum(row_accumulator, col_accumulator)
        # Expected preconditioned gradient, momentum, and parameter.
        if beta > 0.:
            accumulator = beta * accumulator + (1. - beta) * numpy.square(grad)
        else:
            accumulator = accumulator + numpy.square(grad)
        # Update SM3 accumulators.
        row_accumulator = numpy.amax(accumulator, axis=1, keepdims=True)
        col_accumulator = numpy.amax(accumulator, axis=0, keepdims=True)
        exp_p_grad = grad / numpy.sqrt(accumulator)
        gbar = momentum * gbar + (1. - momentum) * exp_p_grad
        var = var - lr * gbar
        # Check that variable and momentum are as expected after one step of
        # training.
        _check_values(x.numpy(), var)
        if momentum > 0.:
            _check_values(opt.state[x]['momentum_buffer'].numpy(), gbar)

@pytest.mark.parametrize(
    'lr, beta',
    [
        [0.1, 0.0],
        [0.1, 0.1],
        [0.2, 0.3],
        [0.3, 0.1],
        [0.1, 0.9]
    ]
)
def test_sparse_updates(lr, beta):
    var = numpy.array([[0.5, 0.05], [0.05, 1.0], [0.15, 3.0], [0.35, 2.0]])
    # A sparse gradient that updates index 1, and 3.
    indices = [1, 3]
    grad_values = numpy.array([0.1, 0.05, 0.01, 1.5])
    grad_indices = torch.LongTensor([[1, 1, 3, 3], [0, 1, 0, 1]])
    shape = torch.Size(var.shape)
    sparse_grad = torch.sparse.FloatTensor(
        grad_indices,
        torch.from_numpy(grad_values),
        shape
    )

    x = _create_tensor(var, sparse_grad)
    opt = SM3([x], lr=lr, beta=beta)
    row_accumulator = numpy.zeros([4, 1])

    for _ in range(5):
        opt.step()

        accumulator = numpy.repeat(row_accumulator, 2, 1)
        if beta > 0.:
            accumulator[indices, :] = beta * accumulator[indices, :] \
                + (1. - beta) * numpy.square(grad_values.reshape(2, 2))
        else:
            accumulator[indices, :] += numpy.square(grad_values.reshape(2, 2))

        row_accumulator = numpy.amax(accumulator, axis=1, keepdims=True)
        # Update SM3 accumulators.
        exp_p_grad_values = grad_values.reshape(2, 2) / numpy.sqrt(accumulator[indices, :])
        var[indices, :] = var[indices, :] - lr * exp_p_grad_values

        _check_values(x.numpy()[indices, :], var[indices, :])
        _check_values(opt.state[x]['accumulator_0'].numpy().flatten(), row_accumulator.flatten())

@pytest.mark.parametrize(
    'lr, momentum, beta',
    [
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.1, 0.0, 0.1],
        [0.2, 0.3, 0.1],
        [0.3, 0.1, 0.2],
        [0.1, 0.2, 0.3]
    ]
)
def test_dense_updates(lr, momentum, beta):
    _test_dense_0d_updates(lr, momentum, beta)
    _test_dense_1d_updates(lr, momentum, beta)
    _test_dense_2d_updates(lr, momentum, beta)


def _check_values(var, exp_var):
    assert numpy.isclose(var, exp_var).all()

def _create_tensor(var, grad):
    # var may change independently of x, so clone is necessary.
    x = torch.from_numpy(var).clone()
    if isinstance(grad, torch.Tensor):
        x.grad = grad
    else:
        x.grad = torch.from_numpy(grad)
    return x
