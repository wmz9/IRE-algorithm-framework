import torch
from copy import deepcopy

class SAMIRE(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rank,  beta, rho, adaptive=False, **kwargs):
        assert rho >= 0.0
        defaults = dict(**kwargs)
        super(SAMIRE, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rank = rank
        self.beta = beta
        self.rho = rho
        self.adaptive = adaptive
        self.init_fisher()
        self.init_mask()
        self.init_last_step_data()
        
    @torch.no_grad()
    def init_fisher(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['fisher'] = torch.zeros_like(p, requires_grad=False).to(p)  

    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).to(p)
    
    @torch.no_grad()
    def init_last_step_data(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['last_step_data'] = torch.zeros_like(p, requires_grad=False).to(p)

    @torch.no_grad()
    def update_mask(self,**kwargs):
        fisher_value_dict = []
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['fisher'] = self.beta * self.state[p]['fisher'] + (1- self.beta) * torch.square(p.grad.data)
                fisher_value_dict.append(self.state[p]['fisher'])

            rank_t = self.rank
        
        # top (1-rank_t) min fisher value
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict])
        keep_num = int(len(fisher_value_list) * rank_t)
        _value, _ = torch.topk(fisher_value_list, keep_num, largest=True)
        threshold = _value[-1]

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = self.state[p]['fisher'] <= threshold.to(p)
                self.state[p]['mask'] = self.state[p]['mask'].float().to(p)
                self.state[p]['mask'].require_grad = False
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0    


    @torch.no_grad()
    def ascent_step(self):
        
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        self.zero_grad()
        self.base_optimizer.zero_grad()
        

    @torch.no_grad()
    def descent_step(self, prog, use_ire):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                self.state[p]['last_step_data'] = p.data.clone()

        self.base_optimizer.step()
        self.zero_grad()
        self.base_optimizer.zero_grad()

        if use_ire:
            for group in self.param_groups:
                for p in group["params"]:
                    g = p.data - self.state[p]['last_step_data']
                    p.data = p.data + g * self.state[p]['mask'] * prog
                    # p.data = self.state[p]['last_step_data'] + g / 2 + g * self.state[p]['mask'] * (prog + 1/2)


    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
