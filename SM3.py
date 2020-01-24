import torch
from torch.optim import Optimizer

class SM3(Optimizer):
    """Implements SM3 algorithm.

    It has been proposed in `Memory-Efficient Adaptive Optimization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        momentum (float, optional): coefficient used to scale prior updates
            before adding (default: 0.0)
        beta (float, optional): coefficient used for exponential moving averages
            (default: 0.0)
    """
    def __init__(self, params, lr=1.0, momentum=0.0, beta=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta: {0}".format(beta))

        defaults = dict(lr=lr, momentum=momentum, beta=beta)
        super(SM3, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            for p in group['params']:
                if p is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise NotImplementedError

                state = self.state[p]
                shape = grad.shape
                rank = len(shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.
                    accumulators = _zero_accumulators(grad.shape, grad.dtype, grad.device)
                    # Add accumulators to state dictionary
                    state.update(accumulators)
                # Get previous accumulators mu_{t-1}
                acc_list = [state[_key(i)] for i in range(rank)]

                # Get update from accumulators
                update = _compute_accumulator(acc_list, shape)
                if beta > 0.:
                    update.mul_(beta)
                update.addcmul_(1. - beta, grad, grad)

                # Update accumulators
                for i in range(rank):
                    nu_max = _max_reduce_except_dim(update, i)
                    state[_key(i)] = torch.max(acc_list[i], nu_max).detach()

                # TODO: Add eps argument
                update.add_(1e-8)

                update.rsqrt_()
                update.mul_(grad)

                if momentum > 0.:
                    update.mul_(1. - momentum)
                    m = state['momentum_buffer']
                    update.add_(momentum, m)
                    state['momentum_buffer'] = update.detach()

                p.data.sub_(group['lr'], update)
                state['step'] += 1
        return loss

def _key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)

def _compute_accumulator(accumulators, shape):
    # Gets the minimum across the accumulators.
    rank = len(shape)
    result = accumulators[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        # Note that torch.min is currently (as of 1.23.2020) not commutative -
        # the order matters for NaN values.
        result = torch.min(result, accumulators[i])
    return result

def _zero_accumulators(shape, dtype, device):
    # Creates initial accumulator
    rank = len(shape)
    accumulator = {}
    for i in range(rank):
        acc_shape = [1]*i + [shape[i]] + [1]*(rank-1-i)
        accumulator[_key(i)] = torch.zeros(acc_shape, dtype=dtype, device=device)
    return accumulator

def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    rank = len(tensor.shape)
    assert dim < rank
    result = tensor
    for d in range(rank):
        if d != dim:
            result = result.max(dim=d, keepdim=True).values
    return result
