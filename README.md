# PyTorch-SM3
 Implements the SM3-II adaptive optimization algorithm for PyTorch.
 The original implementation is in
 [TensorFlow](https://github.com/google-research/google-research/tree/master/sm3)
 and described [here](https://arxiv.org/abs/1901.11150).

 The 'Square-root of Minima of Sums of Maxima of Squared-gradients Method' (SM3)
 algorithm is a memory-efficient adaptive optimization algorithm similar to
 Adam or Adagrad with greatly reduced memory overhead. For an `n x m` matrix,
 Adam uses `O(nm)` memory, while SM3 uses `O(n+m)` due to the chosen cover.
 In general, a tensor of shape `(n_1,, n_2, ..., n_k)` optimized using Adam
 will take `O(prod n_i)` memory while the optimization using SM3 will use
 `O(sum n_i)` memory.

 This advantage drastically shrinks if `momentum > 0.0`. The momentum is
 tracked using a tensor of the same shape as the tensor being optimized. In
 this case, SM3 will use just over half as much memory as Adam.

# Differences
 The TensorFlow version mentions that exponential moving averages can
 be used if desired. I incorporated this into the optimizer. If
 `beta = 0.`, then the accumulated gradient squares is used. If
 `beta > 0.`, then exponential moving averages are used. The authors
 of the paper found that `beta = 0.` was superior for their experiments
 in translation and language models.

# Distilled wisdom from authors
 Their full advice can be seen in the paper and implementation. Here
 are the highlights I captured.
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
