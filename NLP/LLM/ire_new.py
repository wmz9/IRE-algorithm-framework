import torch
import torch.nn.functional as F

##### a simpler and faster version


class IRE_PARAMS(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rank, rank_increase, prog, **kwargs):
        defaults = dict(**kwargs)
        super(IRE_PARAMS, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rank = rank
        self.rank_increase = rank_increase
        self.prog = prog
        self.use_ire = False

    @torch.no_grad()
    def update_mask(self, iter_decay, max_iter_decay, **kwargs):
        if self.rank_increase:
            iter_mod = iter_decay % int(max_iter_decay / 10)
            if iter_mod < max_iter_decay / 20:
                rank_t = self.rank + (0.1 - self.rank) * iter_decay / (max_iter_decay / 20)
            else:
                rank_t = 0.1 + (self.rank - 0.1) * (iter_decay - max_iter_decay / 20) / (max_iter_decay / 20)
        else:
            rank_t = self.rank
    
        fisher_value_dict = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    fisher_value_dict.append(p.grad.abs_())
        
        # top (1-rank_t) min fisher value
        fisher_value_list = torch.cat([x.view(-1) for x in fisher_value_dict])
        keep_num = int(len(fisher_value_list) * (1 - rank_t))
        threshold, _ = torch.kthvalue(fisher_value_list, keep_num)
        del fisher_value_list

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                # assert torch.all(p.grad >= 0)
                    self.state[p]['mask'] = p.grad <= threshold.to(p)
                # assert not self.state[p]['mask'].requires_grad
                # self.state[p]['mask'].require_grad = False

        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         fisher_value_list = torch.square(p.grad.data).square().flatten()
        #         keep_num = int(len(fisher_value_list) * rank_t)
        #         _value, _ = torch.topk(fisher_value_list, keep_num, largest=True)
        #         threshold = _value[-1]
        #         self.state[p]['mask'] = p.grad.data <= threshold
        #         self.state[p]['mask'].require_grad = False
        #         # assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0


    @torch.no_grad()
    def descent_step(self, use_ire=True):
        self.set_descent(use_ire)
        self.step()
    
    @torch.no_grad()
    def step(self, closure = None):
        # if self.use_ire:
        #     for group in self.param_groups:
        #         for p in group["params"]:
        #             if len(self.state[p]) == 0:
        #                 self.state[p]['mask'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        #                 self.state[p]['last_step_data'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        #         group["lr"] = torch.cat([group["lr"] * (1 + self.state[p]['mask'].view(-1) * self.prog) for p in group["params"]])
        # tensor LR support is disateraous in pytorch and hard

        last_step_data = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                        continue
                if len(self.state[p]) == 0:
                    self.state[p]['mask'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=bool)
                last_step_data[p] = p.data.clone()

        self.base_optimizer.step(closure)

        if self.use_ire:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.data - last_step_data[p]
                    # print(p.shape, g.shape, self.state[p]['mask'].shape)
                    g[~self.state[p]['mask']] = 0.
                    p.data = p.data + g * self.prog
                    # p.data = self.state[p]['last_step_data'] + g * 1 / 2 + g * self.state[p]['mask'] * (1 + self.prog - 1 / 2)
        del last_step_data

    def set_descent(self, use_ire=True):
        self.use_ire = use_ire


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none)
        self.base_optimizer.zero_grad(set_to_none)
