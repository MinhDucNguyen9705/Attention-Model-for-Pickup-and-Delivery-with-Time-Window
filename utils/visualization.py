import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, file_path, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False, use_time=True):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    
    # route is one sequence, separating different routes with 0 (depot)
    routes = [r[r != 0] for r in np.split(route.cpu().numpy(), np.where(route == 0)[0]) if (r != 0).any()]
    depot = data['coords'][0, :].cpu().numpy()
    locs = data['coords'][1:, :].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    capacity = demand_scale  # Capacity is always 1

    # Optional time matrix
    tmat = data.get('tmat', None)
    if tmat is not None:
        tmat = tmat.cpu().numpy()

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)

    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects, used_rects, cap_rects, qvs = [], [], [], []
    total_cost = 0.0  # time or distance

    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity

        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)

        cost = 0.0
        cum_demand = 0
        prev_node = 0  # depot index

        # Compute route cost using distance or time
        for node in r:
            if use_time and tmat is not None:
                cost += tmat[prev_node, node]  # travel time
            else:
                x_prev, y_prev = (depot if prev_node == 0 else locs[prev_node - 1])
                x, y = locs[node - 1]
                cost += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            # Visualization patches
            x, y = locs[node - 1]
            d = demands[node - 1]
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            cum_demand += d
            prev_node = node

        # Return to depot
        if use_time and tmat is not None:
            cost += tmat[prev_node, 0]
        else:
            x_prev, y_prev = locs[prev_node - 1]
            cost += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)

        total_cost += cost

        qv = ax1.quiver(
            xs[:-1], ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy', angles='xy', scale=1, color=color,
            label='R{}, # {}, c {} / {}, {} {:.2f}'.format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                't' if use_time else 'd',
                cost
            )
        )
        qvs.append(qv)

    metric = 'time' if use_time else 'distance'
    ax1.set_title(f"{file_path.split('/')[-1]}: {len(routes)} routes, total {metric} {total_cost:.2f}")
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)