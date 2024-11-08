import torch
import torch.nn.functional as F


class AdamWIRE_PARAMS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rank, rank_increase, prog, prog_decay, beta, **kwargs):
        defaults = dict(**kwargs)
        super(AdamWIRE_PARAMS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rank = rank
        self.rank_increase = rank_increase
        self.prog = prog
        self.prog_decay = prog_decay
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
    def update_mask(self, iter_decay, max_iter_decay, track_name_file = None, **kwargs):
        fisher_value_dict = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['fisher'] = self.beta * self.state[p]['fisher'] + (1- self.beta) * torch.square(p.grad.data)
                fisher_value_dict.append(self.state[p]['fisher'])

        if self.rank_increase:
            iter_mod = iter_decay % int(max_iter_decay / 10)
            if iter_mod < max_iter_decay / 20:
                rank_t = self.rank + (0.1 - self.rank) * iter_decay / (max_iter_decay / 20)
            else:
                rank_t = 0.1 + (self.rank - 0.1) * (iter_decay - max_iter_decay / 20) / (max_iter_decay / 20)
        else:
            rank_t = self.rank
        
        # top (1-rank_t) min fisher value
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict])
        if rank_t < 0.5:
            keep_num = int(len(fisher_value_list) * rank_t)
            _value, _ = torch.topk(fisher_value_list, keep_num, largest=True)
            threshold = _value[-1]
        else:
            keep_num = int(len(fisher_value_list) * (1 - rank_t))
            _value, _ = torch.topk(fisher_value_list, keep_num, largest=False)
            threshold = _value[-1]

        if track_name_file is not None:
            name_mask_ratio = {}

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = self.state[p]['fisher'] <= threshold.to(p)
                if track_name_file is not None:
                    name = self.id_map.get(id(p)) or str(id(p))
                    name_mask_ratio[name] = (self.state[p]['mask'].sum() / self.state[p]['mask'].numel()).item()
                self.state[p]['mask'] = self.state[p]['mask'].float().to(p)
                self.state[p]['mask'].require_grad = False
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0

        if track_name_file is not None:
            track_name_file.write(f'{name_mask_ratio}')
            track_name_file.flush()


    @torch.no_grad()
    def descent_step(self, lr, base_lr, use_ire=True):
        self.set_descent(lr, base_lr, use_ire)
        self.step()
    
    @torch.no_grad()
    def step(self, closure = None):
        lr, base_lr, use_ire= self.lr, self.base_lr, self.use_ire

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
                    if self.prog_decay:
                        p.data = p.data + g * self.state[p]['mask'] * self.prog * lr / base_lr
                        # p.data = p.data + g * self.state[p]['mask'] * self.prog * 3 * base_lr / (base_lr + 2 * lr)
                    else:
                        p.data = p.data + g * self.state[p]['mask'] * self.prog
                        # p.data = self.state[p]['last_step_data'] + g * 1 / 2 + g * self.state[p]['mask'] * (1 + self.prog - 1 / 2)

    def set_descent(self, lr, base_lr, use_ire=True):
        self.lr = lr
        self.base_lr = base_lr
        self.use_ire = use_ire


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def init_id_map(self, named_parameters):
        self.id_map = {id(parameter): name for name, parameter in named_parameters}
