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
        arrival_time = self.current_time + travel_time
        begin_service = torch.max(arrival_time, self.tw[self.ids, selected, 0:1])
        current_time = begin_service + self.service[self.ids, selected]

        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            curr_visited = self.curr_visited.scatter(-1, prev_a[:, :, None], 1)
            curr_visited = curr_visited * (prev_a[:, :, None] != 0).float()
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)
            curr_visited = mask_long_scatter(self.curr_visited, prev_a - 1)
            reset_mask = (prev_a[:, :, None] == 0).long() * ((1 << (self.curr_visited.size(-1) * 64)) - 1)
            curr_visited = curr_visited & (~reset_mask)
        
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

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
            curr_visited_loc = self.curr_visited[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, self.demand.size(-1))
            curr_visited_loc = mask_long2bool(self.curr_visited, self.demand.size(-1))
        
        exceeds_cap = (self.demand[self.ids, 1:] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)

        # print('Visited: ', visited_loc)
        # print('Exceed:', exceeds_cap)

        pairs = self.pair.long()[:, 1:]

        delivery_mask = (self.role == -1)[:, 1:]

        # print('Delivery: ', delivery_mask, delivery_mask.shape)

        # delivery_unvisited = curr_visited_loc.squeeze(1)[delivery_mask]
        # print('Delivery Unvisited: ', delivery_unvisited)

        paired_pickup_unv = torch.gather(curr_visited_loc.squeeze(1), dim=1, index=pairs-1).to(delivery_mask.dtype)
        # print('Paired Pickup Unvisited: ', paired_pickup_unv)
        new_delivery_mask = delivery_mask ^ paired_pickup_unv
        new_delivery_mask = new_delivery_mask.unsqueeze(1)
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

        mask_depot = ((self.prev_a == 0) | ~delivery_unvisited.all(dim=1, keepdim=True)) & ((mask_loc == 0).int().sum(-1) > 0)
        # print('Mask Depot: ', self.prev_a == 0, ((mask_loc == 0).int().sum(-1) > 0), ~delivery_unvisited.all(dim=1, keepdim=True), mask_depot)
        mask = torch.cat([mask_depot[:, :, None], mask_loc], dim=-1)

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