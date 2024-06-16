import math

import torch
import torch.nn as nn
import numpy as np


"""
Noise schedules
"""
def vp_beta_schedule(timesteps, b_min, b_max):
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


def linear_beta_schedule(timesteps, b_min, b_max):
    betas = torch.arange(1, timesteps+1) / timesteps
    betas = torch.clip(betas, b_min, b_max)
    return betas


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PCAWhitener(nn.Module):
    def __init__(self, k, pca_params_path):
        super().__init__()

        self.k = k
        params = torch.load(pca_params_path)
        self.register_buffer('mean', params['mean'].float())
        self.register_buffer('components', params['components'].float()[:k])
        self.register_buffer('explained_variance', params['explained_variance'].float()[:k])

    def transform_features(self, features):
        """
        Transform raw trajectory features (N,16*3) to low-dimensional subspace features (N,k)
        """
        features = features.reshape(-1,16*3)
        features = (features - self.mean) @ self.components.T
        features = self.explained_variance**(-.5) * features
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        features = self.explained_variance**(.5) * features
        features = (features @ self.components) + self.mean
        return features


class Whitener(nn.Module):
    def __init__(self, params_path, use_displacements=True, subgoal_idx=None):
        super().__init__()

        self.use_displacements = use_displacements
        self.subgoal_idx = subgoal_idx

        params = torch.load(params_path)
        self.register_buffer('mean', torch.as_tensor(params['mean']).float())
        self.register_buffer('std', torch.as_tensor(params['std']).float())

    def transform_features(self, features):
        """
        Transform raw trajectory features (N,16*3) to low-dimensional subspace features (N,k)
        """
        original_shape = features.shape

        # Whiten
        if self.subgoal_idx is not None:
            features = features.reshape(features.shape[0], 3)
            features = (features - self.mean[self.subgoal_idx]) / self.std[self.subgoal_idx]
        else:
            features = features.reshape(features.shape[0], 16, 3)
            # Compute displacements
            if self.use_displacements:
                features = torch.diff(features, dim=1, prepend=torch.zeros_like(features[:,:1]))
            features = (features - self.mean) / self.std

        features = features.reshape(original_shape)
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        original_shape = features.shape

        # Unwhiten
        if self.subgoal_idx is not None:
            features = features.reshape(features.shape[0], 3)
            features = (features * self.std[self.subgoal_idx]) + self.mean[self.subgoal_idx]
        else:
            features = features.reshape(features.shape[0], 16, 3)
            features = (features * self.std) + self.mean

            # Compute absolute trajectory
            if self.use_displacements:
                features = torch.cumsum(features, dim=1)

        features = features.reshape(original_shape)
        return features


class DummyWhitener(nn.Module):
    def __init__(self, params_path, use_displacements=True):
        super().__init__()

        self.use_displacements = use_displacements

        # params = torch.load(params_path)
        # self.register_buffer('mean', torch.as_tensor(params['mean']).float())
        # self.register_buffer('std', torch.as_tensor(params['std']).float())

    def transform_features(self, features):
        """
        Transform raw trajectory features (N,16*3) to low-dimensional subspace features (N,k)
        """
        original_shape = features.shape
        features = features.reshape(features.shape[0], 16, 3)

        # Compute displacements
        if self.use_displacements:
            features = torch.diff(features, dim=1, prepend=torch.zeros_like(features[:,:1]))

        # Whiten
        # features = (features - self.mean) / self.std

        features = features.reshape(original_shape)
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        original_shape = features.shape

        # Unwhiten
        features = features.reshape(features.shape[0], 16, 3)
        # features = (features * self.std) + self.mean

        # Compute absolute trajectory
        if self.use_displacements:
            features = torch.cumsum(features, dim=1)

        features = features.reshape(original_shape)
        return features
    

class Standardizer(nn.Module):
    def __init__(self, max_dist=50):
        super().__init__()

        self.max_dist = max_dist

    def transform_features(self, ego_agent_features, features):
        features = features.clone()
        features = features.reshape(features.shape[0], -1, 3)
        features[...,:2] = features[...,:2] / self.max_dist
        features[...,2] = features[...,2] / np.pi
        features = features.reshape(features.shape[0], -1)
        return features
        
    def untransform_features(self, ego_agent_features, features):
        features = features.clone()
        features = features.reshape(features.shape[0], -1, 3)
        features[...,:2] = features[...,:2] * self.max_dist
        features[...,2] = features[...,2] * np.pi
        features = features.reshape(features.shape[0], -1)
        return features
    

class VerletStandardizer(nn.Module):
    """
    Standardizes trajectories with Verlet-parameterized actions
    See Section 3.2 in https://arxiv.org/pdf/1905.01296
    """
    
    def __init__(self):
        super().__init__()

        self.max_dist = 4   # magic

    def transform_features(self, ego_agent_features, trajectory):
        trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)
        history = ego_agent_features.ego[:,-2:,:3]

        # Apply Verlet parameterization
        full_trajectory = torch.cat([history, trajectory], dim=1)
        deltas = torch.diff(full_trajectory, dim=1)[:,:-1]
        pred_trajectory = full_trajectory[:,1:-1] + deltas
        actions = full_trajectory[:,2:] - pred_trajectory

        # Standardize actions
        actions = actions / self.max_dist

        actions = actions.reshape(actions.shape[0], -1)
        return actions
        
    def untransform_features(self, ego_agent_features, actions):
        actions = actions.reshape(actions.shape[0], -1, 3)
        history = ego_agent_features.ego[:,-2:,:3]
        
        # Unstandardize actions
        actions = actions * self.max_dist

        # Use Verlet parameterization to calculate trajectory
        states = [history[:,0], history[:,1]]
        for t in range(actions.shape[1]):
            states.append((2 * states[-1]) - states[-2] + actions[:,t])
        trajectory = torch.stack(states[2:], dim=1)

        trajectory = trajectory.reshape(trajectory.shape[0], -1)
        return trajectory


class DiffAndWhitener(nn.Module):
    def __init__(self, norm_val=10):
        super().__init__()

        self.norm_val = norm_val

    def transform_features(self, features):
        features = features.reshape(features.shape[0], 16, 3)
        features = torch.diff(
            features, 
            dim=1,
            prepend=torch.zeros_like(features[:,:1])
        )
        features = features / self.norm_val
        features = features.reshape(features.shape[0], 16 * 3)
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        features = features.reshape(features.shape[0], 16, 3)
        features = features * self.norm_val
        features = torch.cumsum(
            features,
            dim=1
        )
        features = features.reshape(features.shape[0], 16 * 3)
        return features


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """
    Draws samples from an optionally truncated log-logistic distribution.
    Source: https://github.com/intuitive-robots/beso/blob/main/beso/agents/diffusion_agents/k_diffusion/utils.py
    """
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)