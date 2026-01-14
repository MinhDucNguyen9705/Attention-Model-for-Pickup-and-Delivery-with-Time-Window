# API Documentation

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
  - [Model Module](#model-module)
  - [Environment Module](#environment-module)
  - [Baseline Module](#baseline-module)
  - [Training Module](#training-module)
  - [Inference Module](#inference-module)
  - [Utilities](#utilities)
- [Data Structures](#data-structures)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)

---

## Overview

This project implements an attention-based deep reinforcement learning model for solving Capacitated Pickup and Delivery Problems with Time Windows (CPDPTW). The model uses a Graph Attention Encoder-Decoder architecture to generate near-optimal vehicle routes.

---

## Core Components

### Model Module

**File:** `model.py`

#### `class AttentionModel(nn.Module)`

The main attention-based model for solving vehicle routing problems.

**Constructor Parameters:**
- `embed_dim` (int): Dimension of embeddings (default: 128)
- `hidden_dim` (int): Dimension of hidden layers (default: 128)
- `problem` (CPDPTW): Problem instance defining the VRP variant
- `n_encode_layers` (int): Number of encoder layers (default: 2)
- `tanh_clipping` (float): Tanh clipping value for logits (default: 10.0)
- `mask_inner` (bool): Whether to mask inner attention (default: True)
- `mask_logits` (bool): Whether to mask logits (default: True)
- `normalization` (str): Normalization type - 'batch' or 'instance' (default: 'batch')
- `n_heads` (int): Number of attention heads (default: 8)
- `checkpoint_encoder` (bool): Use gradient checkpointing to save memory (default: False)
- `shrink_size` (int): Batch shrinking threshold (default: None)

**Methods:**

##### `forward(input, return_pi=False)`
Forward pass through the model.

**Parameters:**
- `input` (dict): Batch of problem instances containing:
  - `coords`: Node coordinates [batch_size, n_nodes, 2]
  - `demand`: Node demands [batch_size, n_nodes]
  - `tw`: Time windows [batch_size, n_nodes, 2]
  - `service`: Service times [batch_size, n_nodes]
  - `role`: Node roles (-1: pickup, 0: depot, 1: delivery)
  - `pair`: Paired node indices
  - `capacity`: Vehicle capacity
  - `tmat`: Travel time matrix [batch_size, n_nodes, n_nodes]
- `return_pi` (bool): Whether to return the action sequence

**Returns:**
- `cost` (torch.Tensor): Total route cost [batch_size]
- `log_p` (torch.Tensor): Log probabilities of actions
- `pi` (torch.Tensor): Action sequence [batch_size, n_steps] (if return_pi=True)

##### `set_decode_type(decode_type, temp=None)`
Set the decoding strategy.

**Parameters:**
- `decode_type` (str): 'greedy' or 'sampling'
- `temp` (float): Temperature for sampling (default: 1.0)

---

#### `class MultiHeadAttention(nn.Module)`

Multi-head attention mechanism.

**Constructor Parameters:**
- `num_heads` (int): Number of attention heads
- `input_dim` (int): Input dimension
- `embed_dim` (int): Embedding dimension
- `val_dim` (int): Value dimension (default: None, uses embed_dim)
- `key_dim` (int): Key dimension (default: None, uses embed_dim)

**Methods:**

##### `forward(q, h=None, mask=None)`
Compute multi-head attention.

**Parameters:**
- `q` (torch.Tensor): Query tensor [batch_size, n_query, input_dim]
- `h` (torch.Tensor): Key/value tensor [batch_size, n_kv, input_dim] (default: None, uses q)
- `mask` (torch.Tensor): Attention mask [batch_size, n_query, n_kv] (default: None)

**Returns:**
- `out` (torch.Tensor): Attention output [batch_size, n_query, embed_dim]

---

#### `class GraphAttentionEncoder(nn.Module)`

Graph attention encoder for node embeddings.

**Constructor Parameters:**
- `num_heads` (int): Number of attention heads
- `embed_dim` (int): Embedding dimension
- `num_layers` (int): Number of encoder layers
- `node_dim` (int): Input node feature dimension (default: None)
- `normalization` (str): Normalization type (default: 'batch')
- `ff_dim` (int): Feed-forward dimension (default: 512)

**Methods:**

##### `forward(x, mask=None)`
Encode node features.

**Parameters:**
- `x` (torch.Tensor): Node features [batch_size, n_nodes, node_dim]
- `mask` (torch.Tensor): Attention mask (default: None)

**Returns:**
- `h` (torch.Tensor): Node embeddings [batch_size, n_nodes, embed_dim]

---

### Environment Module

**File:** `environment.py`

#### `class CPDPTW(object)`

Capacitated Pickup and Delivery Problem with Time Windows.

**Class Attributes:**
- `VEHICLE_CAPACITY` (float): Default vehicle capacity = 1.0 (normalized)

**Static Methods:**

##### `make_dataset(data_path)`
Create a dataset from file path.

**Parameters:**
- `data_path` (str): Path to directory containing problem instance files

**Returns:**
- `CPDPTWDataset`: Dataset object

##### `make_state(input)`
Initialize problem state.

**Parameters:**
- `input` (dict): Problem instance data

**Returns:**
- `StateCPDPTW`: Initial state

##### `get_costs(input, pi)`
Calculate route costs.

**Parameters:**
- `input` (dict): Problem instance
- `pi` (torch.Tensor): Action sequence

**Returns:**
- `cost` (torch.Tensor): Total route cost
- `mask` (torch.Tensor): Valid action mask

---

#### `class StateCPDPTW(NamedTuple)`

State representation for CPDPTW.

**Fields:**
- `coords` (torch.Tensor): Node coordinates
- `demand` (torch.Tensor): Node demands
- `tw` (torch.Tensor): Time windows
- `service` (torch.Tensor): Service times
- `role` (torch.Tensor): Node roles
- `pair` (torch.Tensor): Paired nodes
- `capacity` (torch.Tensor): Vehicle capacity
- `K` (torch.Tensor): Number of vehicles
- `tmat` (torch.Tensor): Travel time matrix
- `ids` (torch.Tensor): Batch indices
- `prev_a` (torch.Tensor): Previous actions
- `used_capacity` (torch.Tensor): Current vehicle load
- `current_time` (torch.Tensor): Current time
- `visited_` (torch.Tensor): Visited nodes (binary mask)
- `curr_visited` (torch.Tensor): Currently visited in route
- `lengths` (torch.Tensor): Current route lengths
- `cur_coord` (torch.Tensor): Current coordinates
- `i` (torch.Tensor): Step counter

**Methods:**

##### `update(selected)`
Update state after selecting an action.

**Parameters:**
- `selected` (torch.Tensor): Selected node indices [batch_size]

**Returns:**
- `StateCPDPTW`: Updated state

##### `get_mask()`
Get feasibility mask for next action.

**Returns:**
- `mask` (torch.Tensor): Boolean mask [batch_size, 1, n_nodes]

##### `all_finished()`
Check if all instances are finished.

**Returns:**
- `bool`: True if all routes are complete

##### `get_finished()`
Get finished instances.

**Returns:**
- `torch.Tensor`: Boolean tensor indicating finished instances

##### `get_final_cost()`
Calculate final route cost including return to depot.

**Returns:**
- `torch.Tensor`: Final costs [batch_size, 1]

---

#### `class CPDPTWDataset(Dataset)`

PyTorch dataset for CPDPTW instances.

**Constructor Parameters:**
- `data_path` (str): Path to directory containing instance files

**Methods:**

##### `__len__()`
Get dataset size.

**Returns:**
- `int`: Number of instances

##### `__getitem__(idx)`
Get a problem instance.

**Parameters:**
- `idx` (int): Instance index

**Returns:**
- `dict`: Problem instance data

---

#### `class VRPNode()`

Represents a node in the VRP problem.

**Constructor Parameters:**
- `idx` (int): Node index
- `x` (float): X coordinate
- `y` (float): Y coordinate
- `demand` (float): Node demand
- `a` (float): Earliest time (time window start)
- `b` (float): Latest time (time window end)
- `s` (float): Service time
- `role` (int): Node role (-1: pickup, 0: depot, 1: delivery)
- `pair` (int): Paired node index

---

#### `class VRPInstance()`

Represents a complete VRP instance.

**Constructor Parameters:**
- `nodes` (list): List of VRPNode objects
- `capacity` (float): Vehicle capacity
- `K` (int): Maximum number of vehicles
- `tmat` (array): Travel time matrix

**Methods:**

##### `build_tensors()`
Convert instance to PyTorch tensors.

**Returns:**
- `dict`: Instance data as tensors

---

### Baseline Module

**File:** `baseline.py`

#### `class Baseline(object)`

Abstract base class for baselines.

**Methods:**

##### `wrap_dataset(dataset)`
Wrap dataset for baseline evaluation.

##### `unwrap_batch(batch)`
Unwrap batch data.

##### `eval(x, c)`
Evaluate baseline.

##### `get_learnable_parameters()`
Get learnable parameters.

**Returns:**
- `list`: List of parameters

##### `epoch_callback(model, epoch)`
Callback after each epoch.

##### `state_dict()`
Get state dictionary.

**Returns:**
- `dict`: State dictionary

##### `load_state_dict(state_dict)`
Load state dictionary.

---

#### `class RolloutBaseline(Baseline)`

Rollout baseline using a fixed policy.

**Constructor Parameters:**
- `model` (nn.Module): Model to use for baseline
- `problem` (CPDPTW): Problem instance
- `opts` (Namespace): Configuration options
- `epoch` (int): Current epoch (default: 0)

**Methods:**

##### `epoch_callback(model, epoch)`
Challenge baseline with current model and update if improved.

**Parameters:**
- `model` (nn.Module): Current model
- `epoch` (int): Current epoch

---

### Training Module

**File:** `train.py`

#### `get_options()`

Parse command-line arguments for training.

**Returns:**
- `Namespace`: Parsed arguments

**Key Arguments:**
- `--problem`: Problem type (default: 'cvrp')
- `--graph_size`: Problem size (default: 100)
- `--batch_size`: Training batch size (default: 512)
- `--epoch_size`: Instances per epoch (default: 1,280,000)
- `--val_size`: Validation set size (default: 10,000)
- `--data_path`: Path to training data (required)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_dim`: Hidden dimension (default: 128)
- `--n_encode_layers`: Number of encoder layers (default: 3)
- `--n_heads`: Number of attention heads (default: 8)
- `--lr_model`: Learning rate for actor (default: 1e-4)
- `--lr_critic`: Learning rate for critic (default: 1e-4)
- `--lr_decay`: Learning rate decay per epoch (default: 1.0)
- `--n_epochs`: Number of training epochs (default: 10)
- `--baseline`: Baseline type ('rollout', 'critic', 'exponential', or None)
- `--checkpoint_epochs`: Save checkpoint every n epochs (default: 1)
- `--load_path`: Path to load checkpoint
- `--no_cuda`: Disable CUDA
- `--no_tensorboard`: Disable TensorBoard logging

---

### Inference Module

**File:** `infer.py`

#### `get_options()`

Parse command-line arguments for inference.

**Returns:**
- `Namespace`: Parsed arguments

**Key Arguments:**
- `--problem`: Problem type (default: 'pdptw')
- `--graph_size`: Problem size (default: 100)
- `--data_path`: Path to test data (required)
- `--model_path`: Path to trained model checkpoint
- `--output_path`: Path to save results (default: 'logs/results.txt')
- `--output_dir`: Directory for detailed logs (default: 'logs')
- `--batch_size`: Batch size (default: 512)

---

### Utilities

#### Train Utils Module

**File:** `utils/train_utils.py`

##### `rollout(model, dataset, opts)`

Evaluate model on dataset using greedy decoding.

**Parameters:**
- `model` (nn.Module): Model to evaluate
- `dataset` (Dataset): Dataset to evaluate on
- `opts` (Namespace): Configuration options

**Returns:**
- `torch.Tensor`: Route costs for all instances

##### `validate(model, dataset, opts)`

Validate model and print statistics.

**Parameters:**
- `model` (nn.Module): Model to validate
- `dataset` (Dataset): Validation dataset
- `opts` (Namespace): Configuration options

**Returns:**
- `float`: Average cost

##### `train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts)`

Train for one epoch.

**Parameters:**
- `model` (nn.Module): Model to train
- `optimizer` (Optimizer): Optimizer
- `baseline` (Baseline): Baseline for advantage estimation
- `lr_scheduler` (LRScheduler): Learning rate scheduler
- `epoch` (int): Current epoch number
- `val_dataset` (Dataset): Validation dataset
- `problem` (CPDPTW): Problem instance
- `tb_logger` (TbLogger): TensorBoard logger
- `opts` (Namespace): Configuration options

##### `clip_grad_norms(param_groups, max_norm=math.inf)`

Clip gradient norms.

**Parameters:**
- `param_groups` (list): Parameter groups
- `max_norm` (float): Maximum gradient norm

**Returns:**
- `float`: Gradient norm before clipping

##### `move_to(var, device)`

Move variable to device.

**Parameters:**
- `var` (dict/tensor): Variable to move
- `device` (torch.device): Target device

**Returns:**
- Moved variable

---

#### Validation Module

**File:** `utils/validation.py`

##### `validate_file(filename, solution)`

Validate a solution against problem constraints.

**Parameters:**
- `filename` (str): Path to problem instance file
- `solution` (list): List of routes

**Returns:**
- `tuple`: (results, instance) where results contains:
  - `file_name` (str): Instance filename
  - `result` (str): 'VALID' or 'INVALID'
  - `message` (str): Validation message
  - `routes` (int): Number of routes
  - `cost` (float): Total route cost
  - `mean_percent_capacity` (float): Mean capacity utilization
  - `std_percent_capacity` (float): Std dev of capacity utilization
  - `mean_wait` (float): Mean waiting time
  - `std_wait` (float): Std dev of waiting time
  - `feasible` (bool): Feasibility status

##### `convert_solution(tour)`

Convert tour tensor to list of routes.

**Parameters:**
- `tour` (torch.Tensor): Tour sequence

**Returns:**
- `list`: List of routes (each route is a list of node indices)

---

#### Visualization Module

**File:** `utils/visualization.py`

##### `plot_vehicle_routes(data, route, ax, file_path, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False, use_time=True)`

Plot vehicle routes on matplotlib axis.

**Parameters:**
- `data` (dict): Problem instance data
- `route` (torch.Tensor): Route sequence
- `ax` (matplotlib.axes.Axes): Matplotlib axis
- `file_path` (str): Instance file path (for title)
- `markersize` (int): Marker size (default: 5)
- `visualize_demands` (bool): Whether to visualize demands (default: False)
- `demand_scale` (float): Scale factor for demands (default: 1)
- `round_demand` (bool): Whether to round demands (default: False)
- `use_time` (bool): Use time-based cost (default: True)

**Returns:**
- None (modifies ax in-place)

##### `discrete_cmap(N, base_cmap=None)`

Create discrete colormap.

**Parameters:**
- `N` (int): Number of colors
- `base_cmap` (str): Base colormap name (default: None)

**Returns:**
- `Colormap`: Discrete colormap

---

#### Output Logs Module

**File:** `utils/output_logs.py`

##### `log_models(model, opts, output_path, training=True)`

Log model information to file.

**Parameters:**
- `model` (nn.Module): Model to log
- `opts` (Namespace): Configuration options
- `output_path` (str): Path to output file
- `training` (bool): Whether in training mode (default: True)

##### `log_results(results, output_path)`

Log validation results to file.

**Parameters:**
- `results` (tuple): Validation results
- `output_path` (str): Path to output file

##### `log_to_excel(results, file_path, output_dir)`

Log results to Excel file.

**Parameters:**
- `results` (tuple): Validation results
- `file_path` (str): Instance file path
- `output_dir` (str): Output directory

---

## Data Structures

### Problem Instance Dictionary

A problem instance is represented as a dictionary with the following keys:

```python
{
    'coords': torch.Tensor,      # [n_nodes, 2] - node coordinates
    'demand': torch.Tensor,      # [n_nodes] - node demands (positive for pickup, negative for delivery)
    'tw': torch.Tensor,          # [n_nodes, 2] - time windows [earliest, latest]
    'service': torch.Tensor,     # [n_nodes] - service times
    'role': torch.Tensor,        # [n_nodes] - node roles (-1: pickup, 0: depot, 1: delivery)
    'pair': torch.Tensor,        # [n_nodes] - paired node indices
    'capacity': float,           # vehicle capacity (normalized to 1.0)
    'K': int,                    # maximum number of vehicles
    'tmat': torch.Tensor        # [n_nodes, n_nodes] - travel time matrix
}
```

### Instance File Format

Problem instances are stored in text files with the following format:

```
SIZE: <n>
CAPACITY: <cap>
NODES:
<idx> <x> <y> <demand> <earliest_time> <latest_time> <service_time> <pair_idx>
...
EDGES:
<from> <to> <distance>
...
```

---

## Configuration Options

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 128 | Dimension of node embeddings |
| `hidden_dim` | int | 128 | Dimension of hidden layers |
| `n_encode_layers` | int | 3 | Number of encoder layers |
| `n_heads` | int | 8 | Number of attention heads |
| `tanh_clipping` | float | 10.0 | Tanh clipping value |
| `normalization` | str | 'batch' | Normalization type |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 512 | Training batch size |
| `epoch_size` | int | 1,280,000 | Instances per epoch |
| `n_epochs` | int | 10 | Number of training epochs |
| `lr_model` | float | 1e-4 | Learning rate for actor |
| `lr_decay` | float | 1.0 | Learning rate decay per epoch |
| `max_grad_norm` | float | 1.0 | Maximum gradient norm |

### Problem Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | str | 'cvrp' | Problem type |
| `graph_size` | int | 100 | Problem size (number of customers) |
| `data_path` | str | - | Path to data (required) |

---

## Usage Examples

### Training a Model

```python
import torch
from model import AttentionModel
from environment import CPDPTW
from baseline import RolloutBaseline
import torch.optim as optim
from utils.train_utils import train_epoch, validate

# Configuration
class Opts:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 128
    hidden_dim = 128
    n_encode_layers = 3
    n_heads = 8
    tanh_clipping = 10.0
    normalization = 'batch'
    checkpoint_encoder = False
    shrink_size = None
    batch_size = 512
    eval_batch_size = 1024
    lr_model = 1e-4
    lr_decay = 1.0
    data_path = 'data/train/'
    no_progress_bar = False

opts = Opts()

# Initialize problem and model
problem = CPDPTW()
model = AttentionModel(
    embed_dim=opts.embedding_dim,
    hidden_dim=opts.hidden_dim,
    problem=problem,
    n_encode_layers=opts.n_encode_layers,
    tanh_clipping=opts.tanh_clipping,
    mask_inner=True,
    mask_logits=True,
    normalization=opts.normalization,
    n_heads=opts.n_heads,
    checkpoint_encoder=opts.checkpoint_encoder,
    shrink_size=opts.shrink_size
).to(opts.device)

# Initialize baseline and optimizer
baseline = RolloutBaseline(model, problem, opts)
optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

# Load validation dataset
val_dataset = problem.make_dataset(opts.data_path)

# Train for one epoch
train_epoch(model, optimizer, baseline, lr_scheduler, epoch=0, 
            val_dataset=val_dataset, problem=problem, tb_logger=None, opts=opts)
```

### Inference with Trained Model

```python
import torch
from model import AttentionModel
from environment import CPDPTW, CPDPTWDataset
from utils.train_utils import rollout

# Load model
opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
problem = CPDPTW()
model = AttentionModel(
    embed_dim=128,
    hidden_dim=128,
    problem=problem,
    n_encode_layers=3,
    tanh_clipping=10.0,
    mask_inner=True,
    mask_logits=True,
    normalization='batch',
    n_heads=8,
    checkpoint_encoder=False,
    shrink_size=None
).to(opts.device)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pt', map_location=opts.device)
model.load_state_dict(checkpoint['model'])

# Load test dataset
dataset = CPDPTWDataset('data/test/')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)

# Run inference
model.eval()
model.set_decode_type('greedy')
batch = next(iter(dataloader))

with torch.no_grad():
    length, log_p, pi = model(batch, return_pi=True)

print(f"Average route length: {length.mean().item()}")
print(f"Routes: {pi}")
```

### Evaluating a Solution

```python
from utils.validation import validate_file, convert_solution
import torch

# Assume we have a tour tensor from model inference
tour = torch.tensor([0, 1, 2, 0, 3, 4, 0])  # Example tour

# Convert to solution format
solution = convert_solution(tour)

# Validate against instance file
results, instance = validate_file('data/test/instance_001.txt', solution)

file_name, result, message, routes, cost, mean_cap, std_cap, mean_wait, std_wait, feasible = results

print(f"File: {file_name}")
print(f"Result: {result}")
print(f"Message: {message}")
print(f"Number of routes: {routes}")
print(f"Total cost: {cost}")
print(f"Mean capacity utilization: {mean_cap*100:.2f}%")
print(f"Mean waiting time: {mean_wait:.2f}")
```

### Visualizing Routes

```python
import matplotlib.pyplot as plt
from utils.visualization import plot_vehicle_routes
from environment import CPDPTWDataset

# Load instance and solution
dataset = CPDPTWDataset('data/test/')
data = dataset[0]  # Get first instance
tour = torch.tensor([0, 1, 2, 0, 3, 4, 0])  # Example tour

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
plot_vehicle_routes(data, tour, ax, 'instance_001.txt', 
                   visualize_demands=True, demand_scale=50)
plt.savefig('route_visualization.png')
plt.show()
```

### Custom Dataset Creation

```python
from environment import VRPNode, VRPInstance
import torch

# Create nodes
nodes = [
    VRPNode(idx=0, x=0.5, y=0.5, demand=0, a=0, b=100, s=0, role=0, pair=-1),  # Depot
    VRPNode(idx=1, x=0.2, y=0.3, demand=10, a=0, b=50, s=5, role=1, pair=2),   # Pickup
    VRPNode(idx=2, x=0.8, y=0.7, demand=-10, a=10, b=60, s=5, role=-1, pair=1) # Delivery
]

# Create instance
capacity = 50
K = 5  # Max 5 vehicles
tmat = [[0, 10, 15], [10, 0, 12], [15, 12, 0]]  # Travel time matrix

instance = VRPInstance(nodes, capacity, K, tmat)

# Convert to tensors
data = instance.build_tensors()
print(data.keys())  # dict_keys(['coords', 'demand', 'tw', 'service', 'role', 'pair', 'capacity', 'K', 'tmat'])
```

### Batch Processing

```python
from torch.utils.data import DataLoader
from environment import CPDPTWDataset
import torch

# Load dataset
dataset = CPDPTWDataset('data/test/')
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Process batches
results = []
for batch in dataloader:
    with torch.no_grad():
        cost, log_p, pi = model(batch, return_pi=True)
    results.append({
        'cost': cost.cpu(),
        'tours': pi.cpu()
    })

# Aggregate results
all_costs = torch.cat([r['cost'] for r in results])
print(f"Mean cost: {all_costs.mean():.4f}")
print(f"Std cost: {all_costs.std():.4f}")
```

---

## Error Handling

### Common Issues

**1. CUDA Out of Memory**
- Reduce `batch_size` or `eval_batch_size`
- Enable `checkpoint_encoder=True`
- Set `shrink_size` to batch size threshold

**2. Infeasible Solutions**
- Check time window constraints
- Verify pickup-delivery pairing
- Ensure capacity constraints are satisfied

**3. Training Instability**
- Reduce learning rate
- Enable gradient clipping with `max_grad_norm`
- Check baseline configuration

**4. Invalid File Format**
- Ensure instance files follow the correct format
- Check that all required fields are present
- Verify node indices and pairing

---

## Performance Tips

1. **Use CUDA**: Training is significantly faster on GPU
2. **Batch Size**: Larger batches improve GPU utilization but require more memory
3. **Gradient Checkpointing**: Enable for large models to reduce memory usage
4. **Data Loading**: Use `num_workers > 0` for parallel data loading (on systems that support it)
5. **Validation Frequency**: Reduce validation frequency during training to speed up experiments
6. **Model Size**: Adjust `embedding_dim` and `n_encode_layers` based on problem complexity

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{attention_pdptw,
  title={Attention-based Model for Pickup and Delivery with Time Windows},
  author={Your Name},
  year={2026}
}
```

---

## License

See LICENSE file for details.

---

## Contact

For questions or issues, please open an issue on the project repository.
