import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math


class READ(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                loss = (0, 0)
                outputs = (outputs, outputs)

        return outputs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    with autocast():
        # forward
        outputs, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)
    # adapt
    p_sum = outputs.softmax(dim=-1).sum(dim=-2)
    loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()    

    pred = outputs.softmax(dim=-1)
    pred_max = pred.max(dim=-1)[0]
    gamma = math.exp(-1)
    t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
    loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()

    loss = loss_ra - 1 * loss_bal
    
    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        with autocast():
        # forward
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)

    return (outputs, outputs2), (loss_ra.item(), loss_bal.item())


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion_qkv = []
    names_fusion_qkv = []

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)

    return model