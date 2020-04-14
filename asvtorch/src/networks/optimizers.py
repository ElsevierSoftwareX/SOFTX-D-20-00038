import math
import torch
from torch.optim import Optimizer
#from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

class RangerQH(Optimizer):
    r"""Implements the QHAdam optimization algorithm `(Ma and Yarats, 2019)`_.
    Along with Hinton/Zhang Lookahead.
    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter
            groups
        lr (float, optional): learning rate (:math:`\alpha` from the paper)
            (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and its square
            (default: (0.9, 0.999))
        nus (Tuple[float, float], optional): immediate discount factors used to
            estimate the gradient and its square
            (default: (1.0, 1.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability
            (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0.0)
        decouple_weight_decay (bool, optional): whether to decouple the weight
            decay from the gradient-based optimization step
            (default: False)
    Example:
        >>> optimizer = qhoptim.pyt.QHAdam(
        ...     model.parameters(),
        ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        nus=(1.0, 1.0),
        weight_decay=0.0,
        k=6,
        alpha=.5,
        decouple_weight_decay=False,
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "decouple_weight_decay": decouple_weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
        
        #look ahead params
        self.alpha = alpha
        self.k = k         
        

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            decouple_weight_decay = group["decouple_weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("QHAdam does not support sparse gradients")

                

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)
                
                #prep for saved param loading
                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state['step'] = 0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                    #look ahead weight storage now in state dict 
                    param_state['slow_buffer'] = torch.empty_like(p.data)
                    param_state['slow_buffer'].copy_(p.data)
                
                
                param_state['step'] += 1                

                param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)
                
                #integrated look ahead...
                #we do it at the param level instead of group level
                if param_state['step'] % self.k ==0: #group['k'] == 0:
                    slow_p = param_state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor
                

        return loss

    @classmethod
    def _params_to_dict(cls, params):
        return {"lr": params.alpha, "nus": (params.nu1, params.nu2), "betas": (params.beta1, params.beta2)}