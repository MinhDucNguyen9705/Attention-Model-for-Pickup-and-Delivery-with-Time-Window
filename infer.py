import os
import glob
import argparse
import torch
from model import AttentionModel
from environment import CPDPTW, CPDPTWDataset
from utils.train_utils import rollout
from utils.visualization import plot_vehicle_routes
import matplotlib.pyplot as plt

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='pdptw', help="The problem to solve, default 'pdptw'")
    parser.add_argument('--graph_size', type=int, default=100, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file to use for training')
    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                                'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')

    opts, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Unknown args:", unknown)
    return opts

if __name__ == "__main__":
    opts = get_options()
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CPDPTWDataset(opts.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)
    batch = next(iter(dataloader))
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
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None
    )

    state_dict = torch.load(opts.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    model.to(opts.device)
    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi = model(batch, return_pi=True)
    tours = pi

    file_paths = glob.glob(os.path.join(opts.data_path, '*.txt'))

    for i, (data, tour) in enumerate(zip(dataset, tours)):
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(data, tour, ax, file_paths[i], visualize_demands=False, demand_scale=50, round_demand=True)