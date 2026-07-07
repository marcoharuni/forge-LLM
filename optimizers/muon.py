import torch


coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile
def zeropower_polar_express(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for a, b, c in coeffs_list[:steps]:
        A = X @ X.mT
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                if nesterov:
                    g = grad.lerp(buf, momentum)
                else:
                    g = buf
                g = zeropower_polar_express(g, ns_steps)
                g = g.to(dtype=p.dtype)
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g.view_as(p), alpha=-lr * scale)

        return loss
