# PyTorch-SM3
[[source](https://arxiv.org/abs/1901.11150)]
[[TensorFlow](https://github.com/google-research/google-research/tree/master/sm3)]
[[notebook](./SM3_example.ipynb)]

Implements the SM3-II adaptive optimization algorithm for PyTorch.
This algorithm was designed by Rohan Anil, Vineet Gupta, Tomer Koren, and
Yoram Singer and implemented in TensorFlow.

The 'Square-root of Minima of Sums of Maxima of Squared-gradients Method'
(SM3) algorithm is a memory-efficient adaptive optimization algorithm similar
to Adam and Adagrad with greatly reduced memory usage for history tensors.
For an `n x m` matrix, Adam and Adagrad use `O(nm)` memory for history
tensors, while SM3 uses `O(n+m)` due to the chosen cover. In general, a tensor
of shape `(n_1, n_2, ..., n_k)` optimized using Adam will use `O(prod n_i)`
memory for storage tensors, while the optimization using SM3 will use
`O(sum n_i)` memory. Despite storing fewer parameters, this optimization
algorithm manages to be comparably effective.

This advantage drastically shrinks when `momentum > 0`. The momentum is
tracked using a tensor of the same shape as the tensor being optimized. With
momentum, SM3 will use just over half as much memory as Adam, and a bit more
than Adagrad.

If the gradient is sparse, then the optimization algorithm will use `O(n_1)`
memory as there is only a row cover. The value of `momentum` is ignored in
this case.

## Installing
To install with `pip`, you can use `pip install torch-SM3`. Alternatively,
clone the repository and run `python setup.py sdist` and install using the
generated source package.

## Usage
After installing, import the optimizer using `from SM3 import SM3`. The `SM3`
optimizer that is imported can be used exactly the same way a PyTorch
optimizer. For example, the optimizer can be constructed using
`opt = SM3(model.parameters())` with parameter updates being applied using
`opt.step()`.

## Implementation Differences
The algorithm presented by the original authors mentions that the optimization
algorithm can be modified to use exponential moving averages. I incorporated
this into the optimizer. If `beta = 0`, then the accumulated gradient squares
method (i.e. the default SM3 method) is used. If `beta > 0`, then the updates
use exponential moving averages instead. The authors found that `beta = 0` 
was superior for their experiments in translation and language models.

## Requirements
The requirements given in `requirements.txt` are not the absolute minimum -
the optimizer may function for earlier versions of PyTorch than 1.4. However,
these versions are not tested against. Furthermore, a change in the backend
`C++` signatures means that the current version of this package may not run
against earlier versions of PyTorch.

# Wisdom from authors
Their full advice can be seen in the sources above. Here are two points they
emphasize and how to incorporate them.

## Learning rate warm-up
They prefer using a learning rate that quadratically ramps up to the
full learning rate. This is done in the notebook linked above by using the
`LambdaLR` class. After creating the optimizer, you can use the following:
```
lr_lambda = lambda epoch: min(1., (epoch / warm_up_epochs) ** 2)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```
The authors advocate for this as they found that the early gradients were
typically very large in magnitude. By using a warm-up, the accumulated
gradients are not dominated by the first few updates. After this warm-up,
they do not adjust the learning rate.

## Learning rate decay
[Polyak averaging](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
can be useful for training models as the moving average of the parameters
can produce better results than the parameters themselves. As this can be
costly in memory, an alternative they present is to ramp the learning rate
decay to 0 in the last 10% of steps. This can also be achieved using the
`LambdaLR` class with the following `lambda` function:
```
lr_lambda = lambda epoch: min(1., (total_epochs - epoch) / (0.1 * total_epochs))
```
To incorporate both warm-up and decay, we can combine the two functions:
```
lr_lambda = lambda epoch: min(1., (epoch / (warm_up_epochs)) ** 2, (epochs - epoch) / (0.1 * epochs))
```
