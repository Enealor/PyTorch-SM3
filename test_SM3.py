import torch, numpy
from SM3 import SM3, _compute_accumulator
import pytest

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
def test_updates(lr, momentum, beta):
    var = numpy.array([[.5, .5], [.5, .5]], dtype=numpy.float32)
    grad = numpy.array([[.1, .05], [.02, .03]], dtype=numpy.float32)

    x = torch.from_numpy(var).clone()
    x.grad = torch.from_numpy(grad)

    opt = SM3([x], lr=lr, momentum=momentum, beta=beta)

    row_accumulator = numpy.zeros([2, 1])
    col_accumulator = numpy.zeros([1, 2])
    gbar = numpy.zeros_like(grad)
    accumulator = numpy.zeros_like(gbar)
    for i in range(5):
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
        exp_p_grad = grad / numpy.sqrt(accumulator+1e-8)
        gbar = momentum * gbar + (1. - momentum) * exp_p_grad
        var = var - lr * gbar
        # Check that variable and momentum are as expected after one step of
        # training.

        assert numpy.isclose(x.numpy(), var).all(), 'Failed at {0}'.format(i)
        if momentum > 0.:
            assert numpy.isclose(opt.state[x]['momentum_buffer'].numpy(), gbar).all()
