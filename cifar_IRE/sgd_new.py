import torch
from copy import deepcopy


class SGDE(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rank,  beta, **kwargs):
        defaults = dict(**kwargs)
        super(SGDE, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rank = rank
        self.beta = beta
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
    def update_mask(self, **kwargs):
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
    def descent_step(self, prog, use_ire):

        for group in self.param_groups:
            for p in group["params"]:
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
    

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
