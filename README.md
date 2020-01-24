# PyTorch-SM3
 Implements the SM3-II adaptive optimization algorithm for PyTorch.
 The original implementation is in
 [TensorFlow](https://github.com/google-research/google-research/tree/master/sm3)
 and described [here](https://arxiv.org/abs/1901.11150).

 The 'Square-root of Minima of Sums of Maxima of Squared-gradients Method' (SM3)
 algorithm is a memory-efficient adaptive optimization algorithm similar to
 Adam or Adagrad with greatly reduced memory overhead. For an `n x m` matrix,
 Adam uses `O(nm)` memory, while SM3-II uses `O(n+m)` due to the chosen cover.
