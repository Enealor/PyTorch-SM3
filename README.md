# PyTorch-SM3
 [[TensorFlow](https://github.com/google-research/google-research/tree/master/sm3)]
 [[source](https://arxiv.org/abs/1901.11150)]

 Implements the SM3-II adaptive optimization algorithm for PyTorch.
 The original implementation is in TensorFlow.

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

# Differences
 The version presented by the original authors mentions that the optimization
 algorithm can be modified to use exponential moving averages. I incorporated
 this into the optimizer. If `beta = 0`, then the accumulated gradient squares
 method (i.e. the default SM3 method) is used. If `beta > 0`, then the updates
 use exponential moving averages instead. The authors found that `beta = 0` 
 was superior for their experiments in translation and language models.

# Distilled wisdom from authors
 Their full advice can be seen in the sources above. Here are two points they
 emphasize.
## Learning rate warm-up
 They prefer using a learning rate that quadratically ramps up to the
 full learning rate. As I understand, this is done using
 `lr = full_lr * torch.min(1.0, (current_step / warm_up_steps) ** 2).`
 After this, they do not adjust the learning rate.

## Learning rate decay
 [Polyak averaging](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
 can be useful for training models as the moving average of the
 parameters can produce better results than the parameters themselves.
 As this can be costly in memory, an alternative they present is to
 ramp the learning rate decay to 0 in the last 10% of steps.
