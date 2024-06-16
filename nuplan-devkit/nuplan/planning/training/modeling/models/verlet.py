import numpy as np
import torch


def transform(history, trajectory):
    full_trajectory = torch.cat([history, trajectory], dim=1)
    deltas = torch.diff(full_trajectory, dim=1)[:,:-1]
    pred_trajectory = full_trajectory[:,1:-1] + deltas
    residuals = full_trajectory[:,2:] - pred_trajectory
    return residuals


def untransform(history, actions):
    states = [history[:,0], history[:,1]]
    for t in range(actions.shape[1]):
        states.append((2 * states[-1]) - states[-2] + actions[:,t])
    states = torch.stack(states, dim=1)
    return states


if __name__ == '__main__':
    # (B,5+16,3)
    data = np.load('/zfsauton/datasets/ArgoRL/brianyan/trajectories.npy')
    data = torch.as_tensor(data)

    # Only need 1 history frame
    history = data[:,3:5]       # (B,2,3)
    trajectory = data[:,5:]     # (B,16,3)

    actions = transform(history, trajectory)

    with torch.enable_grad():
        actions.requires_grad_(True)
        recon_trajectory = untransform(history, actions)
        loss = torch.norm(recon_trajectory)
        loss.backward()
        import pdb; pdb.set_trace()
        raise

    import matplotlib.pyplot as plt
    actions = actions.reshape(-1,3)
    plt.scatter(actions[:,0], actions[:,1])
    plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')
