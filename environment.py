import os
import glob
import torch
from torch.utils.data import Dataset

class VRPNode():
    def __init__(self, idx, x, y, demand, a, b, s, role, pair):
        super().__init__()
        self.idx = idx
        self.x = x
        self.y = y
        self.demand = demand
        self.a = a  # earliest time
        self.b = b  # latest time
        self.s = s  # service time
        self.role = role  # role of the node (-1: pickup, 0: depot, 1: delivery)
        self.pair = pair  # paired node (pickup-delivery)

class VRPInstance():
    def __init__(self, nodes, capacity, K, tmat):
        super().__init__()
        self.nodes = nodes
        self.capacity = capacity  # vehicle capacity
        self.K = K  # max number of vehicles
        self.tmat = tmat  # [n, n] travel time matrix
    
    def build_tensors(self):
        n = len(self.nodes)
        coords = torch.tensor([[node.x, node.y] for node in self.nodes], dtype=torch.float32)
        demand = torch.tensor([node.demand for node in self.nodes], dtype=torch.float32)
        tw = torch.tensor([[node.a, node.b] for node in self.nodes], dtype=torch.float32)
        service = torch.tensor([node.s for node in self.nodes], dtype=torch.float32)
        role = torch.tensor([node.role for node in self.nodes], dtype=torch.int64)
        pair = torch.tensor([node.pair for node in self.nodes], dtype=torch.int64)

        return dict(
            coords=coords,
            demand=demand,
            tw=tw,
            service=service,
            role=role,
            pair=pair,
            capacity=self.capacity,
            K=self.K,
            tmat=torch.tensor(self.tmat, dtype=torch.float32)
        )
    
from typing import NamedTuple

def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]


def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0

def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)

def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))

class StateCPDPTW(NamedTuple):
    coords: torch.Tensor
    demand: torch.Tensor
    tw: torch.Tensor
    service: torch.Tensor
    role: torch.Tensor
    pair: torch.Tensor
    capacity: torch.Tensor
    K: torch.Tensor
    tmat: torch.Tensor

    ids: torch.Tensor

    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    current_time: torch.Tensor
    visited_: torch.Tensor
    curr_visited: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor

    VEHICLE_CAPACITY = 1.0

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, self.demand.size(-1))
    
    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            current_time=self.current_time[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            curr_visited=self.curr_visited[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        coords = input['coords']
        demand = input['demand']
        tw = input['tw']
        service = input['service']
        role = input['role']
        pair = input['pair']
        capacity = input['capacity']
        K = input['K']
        tmat = input['tmat']

        batch_size, n_nodes, _ = coords.size()

        return StateCPDPTW(
            coords=coords,
            demand=demand,
            tw=tw,
            service=service,
            role=role,
            pair=pair,
            capacity=capacity,
            K=K,
            tmat=tmat,
            ids=torch.arange(batch_size, dtype=torch.int64, device=coords.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=coords.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            current_time=torch.zeros(batch_size, 1, dtype=torch.long, device=coords.device),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_nodes,
                    dtype=visited_dtype, device=coords.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_nodes + 62) // 64, dtype=torch.int64, device=coords.device)
            ),
            curr_visited=(
                torch.zeros(
                    batch_size, 1, n_nodes,
                    dtype=visited_dtype, device=coords.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_nodes + 62) // 64, dtype=torch.int64, device=coords.device)
            ),
            lengths=torch.zeros(batch_size, 1, device=coords.device),
            cur_coord=coords[:, 0, :],
            i=torch.zeros(1, dtype=torch.long, device=coords.device),
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1, keepdim=True)

    def update(self, selected):

        selected = selected[:, None]
        prev_a = selected
        n_nodes = self.demand.size(-1) - 1

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        selected_demand = self.demand[self.ids, torch.clamp(prev_a, 0, n_nodes)]

        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()
        travel_time = self.tmat[self.ids, self.prev_a, prev_a]
        # print('travel_time shape: ', travel_time.shape)
        arrival_time = self.current_time + travel_time
        # print('arrival_time shape: ', arrival_time.shape)
        # print('self.tw shape: ', self.tw[self.ids, selected, 0:1].shape)
        begin_service = torch.max(arrival_time[:, None, :], self.tw[self.ids, selected, 0:1])
        current_time = begin_service.squeeze(1) + self.service[self.ids, selected]

        # print('service shape: ', self.service.shape)
        # print('begin_service shape: ', begin_service.shape)
        # print('current_time shape: ', self.current_time.shape)

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            curr_visited = self.curr_visited.scatter(-1, prev_a[:, :, None], 1)
            curr_visited = curr_visited * (prev_a[:, :, None] != 0).float()
            current_time = current_time * (prev_a != 0).float()
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
            curr_visited = mask_long_scatter(self.curr_visited, prev_a - 1)
            reset_mask = (prev_a[:, :, None] == 0).long() * ((1 << (self.curr_visited.size(-1) * 64)) - 1)
            curr_visited = curr_visited & (~reset_mask)
            current_time = current_time * (prev_a != 0).float()
        
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, current_time=current_time,
            visited_=visited_, curr_visited=curr_visited, lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )
    
    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):

        device = self.coords.device

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
            curr_visited_loc = self.curr_visited[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, self.demand.size(-1))
            curr_visited_loc = mask_long2bool(self.curr_visited, self.demand.size(-1))
        
        exceeds_cap = (self.demand[self.ids, 1:] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # print('Demand: ', self.demand[self.ids, 1:])
        # print('Used Capacity: ', self.used_capacity[:, :, None])
        # print('Sum: ', self.demand[self.ids, 1:] + self.used_capacity[:, :, None])
        # print('Exceed:', exceeds_cap)

        pairs = self.pair.long()[:, 1:]

        delivery_mask = (self.role == -1)[:, 1:]

        # print('Delivery: ', delivery_mask, delivery_mask.shape)

        # delivery_unvisited = curr_visited_loc.squeeze(1)[delivery_mask]
        # print('Delivery Unvisited: ', delivery_unvisited)

        paired_pickup_unv = torch.gather(curr_visited_loc.squeeze(1), dim=1, index=pairs-1).to(delivery_mask.dtype)
        # print('Paired Pickup Unvisited: ', paired_pickup_unv)
        # new_delivery_mask = delivery_mask ^ paired_pickup_unv
        # print((delivery_mask & (~paired_pickup_unv)).unsqueeze(1))
        # new_delivery_mask = new_delivery_mask.unsqueeze(1)
        new_delivery_mask = (delivery_mask & (~paired_pickup_unv)).unsqueeze(1)
        # print('Updated Delivery Mask: ', new_delivery_mask.to(delivery_mask.dtype), new_delivery_mask.shape)
        # print('Mask Loc before update: ', mask_loc)
        # print('Visited_loc shape: ', visited_loc.shape)
        # print('Exceeds_cap shape: ', exceeds_cap.shape)
        # print('New_delivery_mask shape: ', new_delivery_mask.shape)
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | new_delivery_mask.to(delivery_mask.dtype)
        # print('Mask Loc: ', mask_loc, mask_loc.shape)
        # delivery_unvisited = mask_loc[delivery_mask.unsqueeze(1).expand_as(mask_loc)]
            
        delivery_unvisited = mask_loc[delivery_mask.unsqueeze(1).expand_as(mask_loc)]  # shape: (total_true,)
        delivery_unvisited = delivery_unvisited.view(mask_loc.shape[0], -1)
        # print('Delivery Unvisited: ', delivery_unvisited, delivery_unvisited.shape)

        #Check time window constraints
        batch_size, n_loc = mask_loc.shape[0], mask_loc.shape[2]
        travel_time_to_v = self.tmat[self.ids, self.prev_a, :]                      # [B,1,Np1]
        arrival_to_v = self.current_time[:, None, :] + travel_time_to_v             # [B,1,Np1]
        # print('self.current_time shape: ', self.current_time.shape)
        # print('travel_time_to_v shape: ', travel_time_to_v.shape)
        a_v = self.tw[self.ids, torch.arange(n_loc+1), 0][..., None, :]  # [B,1,Np1]
        b_v = self.tw[self.ids, torch.arange(n_loc+1), 1][..., None, :]  # [B,1,Np1]

        arrive_ok = (arrival_to_v <= b_v)                               # [B,1,Np1]
        # print('arrival_to_v shape: ', arrival_to_v.shape)
        # print('a_v shape: ', a_v.shape)
        begin_service_v = torch.maximum(arrival_to_v, a_v)              # [B,1,Np1]
        service_v = self.service[self.ids, torch.arange(n_loc+1)][..., None, :]  # [B,1,Np1]
        time_after_v = begin_service_v + service_v
        # print('Time After V: ', time_after_v, time_after_v.shape)
        # print('service_v shape: ', service_v.shape)
        # print('begin_service_v shape: ', begin_service_v.shape)
        # print('time_after_v shape: ', time_after_v.shape)

        # If v is a pickup, include its delivery's travel + wait to a_d + service_d
        is_v_pickup = (self.role == 1)                                            # [B,Np1]
        d_of_v = self.pair                                                        # [B,Np1]
        has_valid_d = (d_of_v > 0) & is_v_pickup                             # [B,Np1]

        # Build per-(b,v) indexing to get t(v->d), a_d, b_d, service_d
        bf = torch.arange(batch_size)[:, None].expand(batch_size, n_loc + 1)          # [B,Np1]
        v_idx = torch.arange(n_loc+1)[None, :].expand(batch_size, n_loc + 1)                              # [B,Np1]
        d_idx = d_of_v.clamp(min=0)                                          # [B,Np1]

        t_v_d = self.tmat[bf, v_idx, d_idx]                                       # [B,Np1]
        a_d = self.tw[bf, d_idx, 0]                                               # [B,Np1]
        b_d = self.tw[bf, d_idx, 1]                                               # [B,Np1]
        service_d = self.service[bf, d_idx]                                       # [B,Np1]

        arrival_d = time_after_v.squeeze(1) + t_v_d                          # [B,Np1]
        begin_d = torch.maximum(arrival_d, a_d)                              # [B,Np1]
        complete_d = begin_d + service_d                                     # [B,Np1]
        # print('arrival_d shape: ', arrival_d.shape)
        # print('a_d shape: ', a_d.shape)
        # print('begin_d shape: ', begin_d.shape)
        # print('complete_d shape: ', complete_d.shape)

        # Apply pickup adjustment only where valid (keep original time for others)
        delta_pick = (complete_d - time_after_v.squeeze(1)) * has_valid_d    # [B,Np1]
        time_after_v_adj = time_after_v + delta_pick[:, None, :]             # [B,1,Np1]
        
        # print('Time After V Adj: ', time_after_v_adj, time_after_v_adj.shape)
        # print(d_idx[has_valid_d])

        # for i in range(batch_size):
        #     remaining_delivery_indices = torch.where(paired_pickup_unv[i])[0]+1
        #     # Append d_idx to remaining_delivery_indices
        #     remaining_delivery_indices = torch.cat([remaining_delivery_indices, d_idx[has_valid_d]], dim=0)
        #     print('Batch ', i, ' Remaining Delivery Indices: ', remaining_delivery_indices)

        visited_full = (self.visited_ if self.visited_.dtype == torch.uint8 else mask_long2bool(self.visited_, n_loc+1)).bool()
        curr_vis_full = (self.curr_visited if self.visited_.dtype == torch.uint8 else mask_long2bool(self.curr_visited, n_loc+1)).bool()
        visited = visited_full[:, :, 1:]                                     # [B,1,N]
        curr_vis = curr_vis_full[:, :, 1:]                                   # [B,1,N]

        is_del_full = (self.role == -1).to(device)                                           # [B,Np1]
        is_del = is_del_full[:, 1:]                                          # [B,N]
        not_visited = ~visited                                               # [B,1,N]
        pairs_non_depot = self.pair[:, 1:]                                        # [B,N]
        paired_pick_for_del = torch.gather(curr_vis.squeeze(1), 1, pairs_non_depot - 1).bool()  # [B,N]
        open_del_before = (is_del & paired_pick_for_del & not_visited.squeeze(1))               # [B,N]

        open_after_v = open_del_before.unsqueeze(1).expand(batch_size, n_loc+1, n_loc).clone() # [B,Np1,N]

        # remove delivery j if v == j and v is a delivery
        idx_all = torch.arange(n_loc+1).to(device)
        idx_1_to_N = torch.arange(1, n_loc+1).to(device)
        v_eq_j_mask = (idx_1_to_N.unsqueeze(0).unsqueeze(0) == idx_all.unsqueeze(0).unsqueeze(2))  # [1,Np1,N]
        remove_mask = v_eq_j_mask.to(device) & is_del_full.unsqueeze(-1).to(device)                # [B,Np1,N]
        open_after_v = open_after_v.to(device) & (~remove_mask).to(device)
        post_node = torch.where(has_valid_d.to(device), d_of_v.to(device), idx_all.to(device)).to(device)

        # add paired delivery of v if v is pickup
        one_hot_j = torch.zeros(batch_size, n_loc+1, n_loc, dtype=torch.bool).to(device)
        for b in range(batch_size):
            vs = torch.nonzero(has_valid_d[b], as_tuple=False).squeeze(-1).to(device)
            if vs.numel() > 0:
                js = (d_of_v[b, vs] - 1).clamp(min=0)
                one_hot_j[b, vs, js] = True
        open_after_v = open_after_v.to(device) | one_hot_j.to(device)
        # print('Open after v', open_after_v[:, 0, :], open_after_v.shape)

        # print(idx_all.shape, idx_1_to_N.shape)
        B = len(self.ids)
        N_all = idx_all.shape[0]       # 101
        N_1_to_N = idx_1_to_N.shape[0] # 100

        # batch_idx = self.ids[:, None].expand(B, N_all, N_1_to_N)
        # from_idx  = idx_all[None, :, None].expand(B, N_all, N_1_to_N)
        # to_idx    = idx_1_to_N[None, None, :].expand(B, N_all, N_1_to_N)
        # print(batch_idx.shape, from_idx.shape, to_idx.shape)
        # t_from_v_to_j = self.tmat[batch_idx, from_idx, to_idx]
        t_from_post_to_all = self.tmat[self.ids.expand(B, n_loc+1), post_node, :].to(device)
        t_from_v_to_j = t_from_post_to_all[:, :, 1:].to(device)
        # print('T from V to J: ', t_from_v_to_j[:, 0, :], t_from_v_to_j.shape)
        # print('T from V to J: ', t_from_v_to_j[:, 19, :], t_from_v_to_j.shape)
        # t_from_v_to_j = self.tmat[self.ids, idx_all[None, :, None], idx_1_to_N[None, None, :]]

        a_j = self.tw[self.ids, idx_1_to_N, 0][:, None, :].to(device)                   # [B,1,N]
        b_j = self.tw[self.ids, idx_1_to_N, 1][:, None, :].to(device)                   # [B,1,N]
        # service_j = self.service[self.ids, idx_1_to_N][:, None, :]

        # arrival to j and begin_service_j considering earliest time
        # print('Time after v adj shape: ', time_after_v_adj, time_after_v_adj.transpose(1, 2).shape)   
        # print('t from v to j shape: ', t_from_v_to_j[:, 19, :], t_from_v_to_j.shape)
        arrival_vj = time_after_v_adj.transpose(1, 2)  + t_from_v_to_j                   # [B,Np1,N]
        # print('Arrival VJ: ', arrival_vj[:, 0, :], arrival_vj.shape)
        # print('Arrival VJ: ', arrival_vj[:, 19, :], arrival_vj[:, 19, :].shape)
        # print('a_j shape: ', a_j.shape, a_j[:, None, :].shape)
        begin_vj = torch.maximum(arrival_vj, a_j)           # [B,Np1,N]
        # print('Begin VJ: ', begin_vj[:, 0, :], begin_vj.shape)
        # print('Begin VJ: ', begin_vj[:, 19, :], begin_vj[:, 19, :].shape)
        # print('b_j shape: ', b_j, b_j.shape)
        # print(self.tw[self.ids, :, 1])

        # Feasible if we can START by b_j (standard PDPTW: b is latest start)
        feas_open_after = (begin_vj <= b_j)                 # [B,Np1,N]
        # print('Feas Open After: ', feas_open_after[:, 0, :], feas_open_after.shape)
        # print('Feas Open After: ', feas_open_after[:, 19, :], feas_open_after.shape)
        all_open_ok = torch.where(open_after_v, feas_open_after, torch.ones_like(feas_open_after)).all(dim=-1, keepdim=True)  # [B,Np1,1]

        # Local feasibility at v: arrival_to_v must be ≤ b_v as usual
        tw_ok_v = (arrival_to_v <= b_v) & all_open_ok.transpose(1, 2)
        # print('arrival to v <= b_v: ',(arrival_to_v <= b_v))
        # print('all open ok: ', all_open_ok.transpose(1, 2), all_open_ok.transpose(1, 2).shape)
        # print('TW OK V before depot check: ', tw_ok_v, tw_ok_v.shape)

        B = self.demand.size(0)
        Np1 = self.demand.size(-1)
        N = Np1 - 1
        device = self.demand.device

        idx_all = torch.arange(Np1, device=device)        # 0..N
        idx_1_N = torch.arange(1, Np1, device=device)     # 1..N

        # Depot window
        a0 = self.tw[self.ids, 0, 0]    # [B]
        b0 = self.tw[self.ids, 0, 1]    # [B]

        # Per-delivery data
        a_j_full = self.tw[self.ids, idx_1_N, 0]          # [B,N]
        b_j_full = self.tw[self.ids, idx_1_N, 1]          # [B,N]
        svc_j_full = self.service[self.ids, idx_1_N]      # [B,N]

        # Open deliveries after choosing each v
        S = open_after_v.clone()                          # [B,Np1,N] bool

        # State per (b,v)
        T = time_after_v_adj.squeeze(1).clone()           # [B,Np1]
        # prev_node = idx_all[None, :].expand(B, Np1).clone()   # start at v
        prev_node = post_node.clone()                          # start at post(v)
        batch_idx_bv = torch.arange(B, device=device)[:, None].expand(B, Np1)

        # Track feasibility per (b,v)
        all_steps_feasible = torch.ones(B, Np1, dtype=torch.bool, device=device)

        # Iterate up to N selections
        for _ in range(N):
            # If nothing left open for some (b,v), we can skip them
            has_open = S.any(dim=-1)                      # [B,Np1]
            if not has_open.any():
                break

            # t(prev -> j) for all j (1..N), per (b,v)
            # Shape: [B,Np1,Np1] then slice to deliveries 1..N -> [B,Np1,N]
            t_prev_to_all = self.tmat[batch_idx_bv, prev_node, :]      # [B,Np1,Np1]
            t_prev_to_j   = t_prev_to_all[:, :, 1:]                    # [B,Np1,N]

            # Expand per-delivery windows/services to [B,Np1,N]
            a_j = a_j_full[:, None, :].expand(B, Np1, N)               # [B,Np1,N]
            b_j = b_j_full[:, None, :].expand(B, Np1, N)               # [B,Np1,N]
            svc_j = svc_j_full[:, None, :].expand(B, Np1, N)           # [B,Np1,N]

            # Arrival / begin / finish
            arrive_j = T[:, :, None] + t_prev_to_j                     # [B,Np1,N]
            begin_j  = torch.maximum(arrive_j, a_j)                    # [B,Np1,N]
            feas_j   = (begin_j <= b_j) & S                            # must be open and time-feasible
            any_feas = feas_j.any(dim=-1)                              # [B,Np1]

            # If a (b,v) has open deliveries but none feasible now -> deadlock for that (b,v)
            deadlock_now = has_open & (~any_feas)
            if deadlock_now.any():
                all_steps_feasible = all_steps_feasible & (~deadlock_now)
                # For these, we won't update T/prev/S anymore; just keep them marked infeasible.
                # To avoid updating them below, mask their feas_j to all False:
                feas_j = torch.where(deadlock_now[:, :, None], torch.zeros_like(feas_j), feas_j)

            # Finish times (INF where infeasible)
            INF = 1e12
            finish_j = torch.where(feas_j, begin_j + svc_j, torch.full_like(begin_j, INF))  # [B,Np1,N]

            # Select the j with minimal finish time (ties arbitrary)
            best_finish, best_pos = finish_j.min(dim=-1)               # [B,Np1], [B,Np1] (pos in 0..N-1)
            # Build the chosen index (1..N), but only where any_feas is True
            chosen_j = (best_pos + 1)                                  # [B,Np1]
            # Update only on those (b,v) where we had any feasible
            upd_mask = any_feas & (~deadlock_now)

            # Gather begin/finish at chosen j for updates
            # (indexing gather needs shape align)
            gather_idx = best_pos.clamp(min=0).unsqueeze(-1)           # [B,Np1,1]
            begin_star = begin_j.gather(-1, gather_idx).squeeze(-1)    # [B,Np1]
            finish_star = (begin_star + svc_j.gather(-1, gather_idx).squeeze(-1))  # [B,Np1]

            # Apply updates
            T = torch.where(upd_mask, finish_star, T)                  # [B,Np1]
            prev_node = torch.where(upd_mask, chosen_j, prev_node)     # [B,Np1]

            # Remove chosen from S
            # Build one-hot at position best_pos for removal
            rem_mask = torch.zeros_like(S, dtype=torch.bool)           # [B,Np1,N]
            rem_mask.scatter_(-1, best_pos.unsqueeze(-1), True)
            S = S & (~rem_mask)                                        # remove chosen j where it was picked

        # After serving all open deliveries (where possible), go to depot
        t_prev_to_0 = self.tmat[batch_idx_bv, prev_node, torch.zeros_like(prev_node)]   # [B,Np1]
        arrive_0 = T + t_prev_to_0
        # print('Arrive 0: ', arrive_0.shape)
        # print('a0 shape: ', a0.shape, a0[:, None].shape)
        begin_0  = torch.maximum(arrive_0, a0)                                  # [B,Np1]
        # print(begin_0.shape, b0.shape)
        # print('begin_0<=b0: ', (begin_0<=b0), begin_0.shape)
        # print('all steps feasible: ', all_steps_feasible, all_steps_feasible.shape)
        depot_ok_exact = (begin_0 <= b0) & all_steps_feasible                   # [B,Np1]

        # Combine with your per-node checks
        # print('depot_ok_exact', depot_ok_exact, depot_ok_exact.shape)
        # print('tw_ok_v before depot check: ', tw_ok_v, tw_ok_v.shape)
        tw_ok_v = tw_ok_v & depot_ok_exact[:, None, :]   # keep [B,1,Np1]
        # print('TW OK V: ', tw_ok_v.shape)
        # print('TW OK V before depot check: ', ~tw_ok_v, tw_ok_v.shape)
        mask_depot = ((self.prev_a == 0) | ~delivery_unvisited.all(dim=1, keepdim=True)) & ((mask_loc == 0).int().sum(-1) > 0)
        # print('Mask Depot: ', self.prev_a == 0, ((mask_loc == 0).int().sum(-1) > 0), ~delivery_unvisited.all(dim=1, keepdim=True), mask_depot)
        # print('Visited loc: ', visited_loc, visited_loc.shape)
        # print('Mask loc final: ', mask_loc, mask_loc.shape)
        # print('TW ok v: ', tw_ok_v[:, :, 1:].shape)
        mask_loc = mask_loc | (~tw_ok_v[:, :, 1:].to(mask_loc.dtype))
        # print('Mask loc final: ', mask_loc, mask_loc.shape)
        # print('Mask depot: ', mask_depot.shape)
        mask = torch.cat([mask_depot[:, :, None], mask_loc], dim=-1)
        # print('Final mask: ', mask, mask.shape)

        return mask

    def construct_solutions(self, actions):
        return actions
    
class CPDPTW(object):

    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        sorted_pi = pi.data.sort(1)[0]

        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], 0),
                dataset['demand'][:, 1:]
            ), dim = 1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range (pi.size(1)):
            used_cap += d[:, i]
            used_cap[used_cap < 0] = 0

        loc_with_depot = torch.cat((dataset['coords'][:, :1], dataset['coords'][:, 1:]), dim=1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['coords'][:, 0]).norm(p=2, dim=1)
            + (d[:, -1] - dataset['coords'][:, 0]).norm(p=2, dim=1)
        ), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return CPDPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCPDPTW.initialize(*args, **kwargs)

class CPDPTWDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        file_paths = glob.glob(os.path.join(data_path,'*.txt'))
        for fp in file_paths:
            instance = self.read_pdptw_file(fp)
            self.data.append(instance)

    def __len__(self):
        return len(self.data)
    
    def read_pdptw_file(self, filepath):
        data = {
            "metadata": {},
            "nodes": [],
            "edges": []
        }
        
        with open(filepath, 'r') as file:
            lines = file.readlines()

        section = "metadata"
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Section switches
            if line == "NODES":
                section = "nodes"
                continue
            elif line == "EDGES":
                section = "edges"
                continue
            elif line == "EOF":
                break

            if section == "metadata":
                if ":" in line:
                    key, value = line.split(":", 1)
                    data["metadata"][key.strip()] = value.strip()

            elif section == "nodes":
                parts = line.split()
                if len(parts) >= 8:
                    node = {
                        "id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "demand": int(parts[3]),
                        "ready_time": int(parts[4]),
                        "due_time": int(parts[5]),
                        "service_time": int(parts[6]),
                        "pickup_or_delivery": 0 if (int(parts[7]) == 0 and int(parts[8]) == 0) else (-1 if int(parts[7]) > 0 else 1),  # 0 = depot, 1 = pickup, -1 = delivery,...
                        "pair_id": int(parts[7]) if int(parts[7]) > 0 else int(parts[8])
                    }
                    data["nodes"].append(node)

            elif section == "edges":
                weights = list(map(int, line.split()))
                data["edges"].append(weights)

        nodes = []

        for node_data in data["nodes"]:
            node = VRPNode(
                idx=node_data["id"],
                x=node_data["x"],
                y=node_data["y"],
                demand=node_data["demand"]/int(data['metadata'].get("CAPACITY", 1)),
                a=node_data["ready_time"],
                b=node_data["due_time"],
                s=node_data["service_time"],
                role=node_data["pickup_or_delivery"],
                pair=node_data["pair_id"]
            )
            nodes.append(node)

        capacity = int(data["metadata"].get("CAPACITY", 0))
        K = int(data["metadata"].get("NUM_VEHICLES", 1e10))
        dmat = data["edges"]
        instance = VRPInstance(nodes, capacity, K, dmat)

        return instance
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance.build_tensors()