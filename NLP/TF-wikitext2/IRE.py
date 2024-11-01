import torch
import torch.nn.functional as F


class AdmWIRE(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, rank, prog, beta, prog_decay=True, **kwargs):
        defaults = dict(**kwargs)
        super(AdmWIRE, self).__init__(model.parameters(), defaults)
        self.model = model
        self.base_optimizer = base_optimizer(self.model.parameters(), **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.prog_decay = prog_decay
        self.rank = rank
        self.prog = prog
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

        # top (1-rank) min fisher value
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict])
        keep_num = int(len(fisher_value_list) * (1 - self.rank))
        _value, _ = torch.topk(fisher_value_list, keep_num, largest=False) 
        threshold = _value[-1]

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = self.state[p]['fisher'] <= threshold.to(p)
                self.state[p]['mask'] = self.state[p]['mask'].float().to(p)
                self.state[p]['mask'].require_grad = False
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0


    @torch.no_grad()
    def descent_step(self, lr, base_lr):

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['last_step_data'] = p.data.clone()

        self.base_optimizer.step()
        self.model.zero_grad()
        self.base_optimizer.zero_grad()

        for group in self.param_groups:
            for p in group["params"]:
                g = p.data - self.state[p]['last_step_data']
                if self.prog_decay:
                    p.data = p.data + g * self.state[p]['mask'] * self.prog * lr / base_lr
                else:
                    p.data = p.data + g * self.state[p]['mask'] * self.prog


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
