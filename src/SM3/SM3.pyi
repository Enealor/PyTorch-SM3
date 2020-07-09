from torch.optim import Optimizer, _params_t

class SM3(Optimizer):
    def __init__(self, params: _params_t, lr: float=..., momentum: float=..., beta: float=..., eps: float=...) -> None: ...
