from typing import cast
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import OmegaConf
import numpy as np
from diffusers import DDIMScheduler

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    UrbanDriverOpenLoopModel,
    convert_predictions_to_trajectory,
)
from nuplan.planning.training.modeling.models.diffusion_utils import (
    SinusoidalPosEmb,
    Standardizer,
    VerletStandardizer
)
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.models.encoder_decoder_layers import (
    ParallelAttentionLayer
)
from nuplan.planning.training.modeling.models.positional_embeddings import VolumetricPositionEncoding2D
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.feature_builders.agent_history_feature_builder import (
    AgentHistoryFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.agent_trajectory_target_builder import (
    AgentTrajectoryTargetBuilder
)


class ConditionalDiffusionModel(UrbanDriverOpenLoopModel):
    def __init__(
        self,

        model_params,
        feature_params,
        target_params,

        T: int = 100,
        predictions_per_sample: int = 4,

        num_encoder_layers: int = 2,
        num_trajectory_decoder_layers: int = 2,
        num_global_decoder_layers: int = 2,

        use_loss_weight: bool = True,
        use_weight_init: bool = True,

        unconditional: bool = False,    # ignores scene tokens
        use_relative: bool = False,     # use relative temporal attention over trajectory features, removes absolute temporal embeddings for traj
        load_checkpoint_path: str = '',

        max_dist: float = 50,           # used to normalize ALL tokens (incl. trajectory)
        use_noise_token: bool = True,   # concat "noise level" token when self-attending, otherwise add to all tokens
        use_positional_encodings: bool = True,

        use_verlet: bool = True         # use Verlet action wrapping
    ):
        super().__init__(model_params, feature_params, target_params)

        self.feature_builders = [
            AgentHistoryFeatureBuilder(
                    trajectory_sampling=feature_params.past_trajectory_sampling,
                    max_agents=feature_params.max_agents
                ),
            VectorSetMapFeatureBuilder(
                map_features=feature_params.map_features,
                max_elements=feature_params.max_elements,
                max_points=feature_params.max_points,
                radius=feature_params.vector_set_map_feature_radius,
                interpolation_method=feature_params.interpolation_method
            )
        ]
        self.target_builders = [
            EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
            AgentTrajectoryTargetBuilder(
                trajectory_sampling=feature_params.past_trajectory_sampling,
                future_trajectory_sampling=target_params.future_trajectory_sampling,
                max_agents=feature_params.max_agents
            )
        ]

        self.feature_dim = model_params.global_embedding_size
        self.output_dim = target_params.future_trajectory_sampling.num_poses * Trajectory.state_size()

        self.T = T
        self.H = target_params.future_trajectory_sampling.num_poses
        
        self.unconditional = unconditional
        self.use_relative = use_relative
        self.predictions_per_sample = predictions_per_sample
        self.num_encoder_layers = num_encoder_layers
        self.num_trajectory_decoder_layers = num_trajectory_decoder_layers
        self.num_global_decoder_layers = num_global_decoder_layers
        self.max_dist = max_dist
        self.use_noise_token = use_noise_token
        self.use_positional_encodings = use_positional_encodings
        self.use_loss_weight = use_loss_weight
        self.use_verlet = use_verlet

        if self.use_verlet:
            self.standardizer = VerletStandardizer()
        else:
            self.standardizer = Standardizer(max_dist=max_dist)

        # Noise level encoder
        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.sigma_proj_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

        # Diffusion model components
        del self.global_head # don't need this

        self.map_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(2)])

        self.encoder_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_encoder_layers)])

        self.trajectory_dim = 3
        self.trajectory_encoder = nn.Linear(self.trajectory_dim, self.feature_dim)
        self.trajectory_time_embeddings = nn.Embedding(self.H, self.feature_dim)

        self.extended_type_embedding = nn.Embedding(2, self.feature_dim) # trajectory, noise token

        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_trajectory_decoder_layers)])

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_global_decoder_layers)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.trajectory_dim)
        )

        self.rel_pos_enc = VolumetricPositionEncoding2D(self.feature_dim)

        # Diffusion
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.T,
            beta_schedule='scaled_linear',
            prediction_type='epsilon',
        )
        self.scheduler.set_timesteps(self.T)

        # Weight initialization
        if use_weight_init:
            self.apply(self._init_weights)

        # Load weights
        if load_checkpoint_path:
            try:
                base_path = '/'.join(load_checkpoint_path.split('/')[:-2])
                config_path = os.path.join(base_path, 'code/hydra/config.yaml')
                model_config = OmegaConf.load(config_path)
                torch_module_wrapper = build_torch_module_wrapper(model_config.model)

                model = LightningModuleWrapper.load_from_checkpoint(
                    load_checkpoint_path, model=torch_module_wrapper
                ).model

                self.load_state_dict(model.state_dict(), strict=True)
                print('Loaded model weights in constructor')
            except Exception as E:
                print('Failed to load model weights in constructor -- this is fine for evaluation')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features: FeaturesType) -> TargetsType:
        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        batch_size = ego_agent_features.batch_size

        scene_features, scene_feature_masks = self.encode_scene_features(ego_agent_features, vector_set_map_data)

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)

        if self.training:
            noise = torch.randn(ego_gt_trajectory.shape, device=ego_gt_trajectory.device)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (ego_gt_trajectory.shape[0],), 
                                        device=ego_gt_trajectory.device).long()

            ego_noisy_trajectory = self.scheduler.add_noise(ego_gt_trajectory, noise, timesteps)

            pred_noise = self.denoise(
                ego_noisy_trajectory, 
                timesteps, 
                scene_features, 
                scene_feature_masks
            )
            
            output = {
                # TODO: fix trajectory, right now we only care about epsilon at training and this is a dummy value
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(pred_noise)),
                "epsilon": pred_noise,
                "gt_epsilon": noise
            }
            return output
        else:
            # Multiple predictions per sample
            scene_features = (
                scene_features[0].repeat_interleave(self.predictions_per_sample,0),
                scene_features[1].repeat_interleave(self.predictions_per_sample,0),
                scene_features[2].repeat_interleave(self.predictions_per_sample,0),
                scene_features[3].repeat_interleave(self.predictions_per_sample,0),
            )
            ego_agent_features = ego_agent_features.repeat_interleave(self.predictions_per_sample,0)
            scene_feature_masks = scene_feature_masks.repeat_interleave(self.predictions_per_sample,0)

            # Sampling / inference
            ego_trajectory = torch.randn((batch_size * self.predictions_per_sample, self.H * 3), device=scene_features[0].device)
            timesteps = self.scheduler.timesteps
            
            intermediate_trajectories = []

            for t in timesteps:
                with torch.no_grad():
                    residual = self.denoise(
                        ego_trajectory, 
                        t.to(ego_trajectory.device),
                        scene_features,
                        scene_feature_masks
                    )

                out = self.scheduler.step(residual, t, ego_trajectory)
                ego_trajectory = out.prev_sample
                ego_trajectory_clean = out.pred_original_sample

                # Intermediate trajectories
                int_ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory_clean)
                intermediate_trajectories.append(int_ego_trajectory)

            ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)

            # Otherwise, who cares TODO later
            ego_trajectory = ego_trajectory.reshape(batch_size, self.predictions_per_sample, self.output_dim)
            best_trajectory = ego_trajectory[:,0]

            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
                "multimodal_trajectories": ego_trajectory,
                "intermediate": intermediate_trajectories
            }
        
    def extract_agent_features(self, agent_history):
        ego_features = agent_history.ego
        agent_features = agent_history.data
        agent_masks = agent_history.mask

        B, T, A, D = agent_features.shape
        device = agent_features.device

        ego_agent_features = torch.cat([ego_features[:,:,None], agent_features], dim=2)
        ego_agent_masks = torch.cat([torch.ones_like(agent_masks[:,:,:1]), agent_masks], dim=2)

        # All of this is just to match the format of the old agent features
        # B x 5 x (A+1) x 7 -> B x (A+1) x 20 (5?) x 8
        ego_agent_features = ego_agent_features.permute(0,2,1,3)
        ego_agent_masks = ego_agent_masks.permute(0,2,1)

        ego_agent_positions = ego_agent_features[:,:,-1,:2]

        ego_agent_features = torch.cat([ego_agent_features, torch.zeros(B, A+1, 15, D, dtype=torch.bool, device=device)], dim=2)
        ego_agent_masks = torch.cat([ego_agent_masks, torch.zeros(B, A+1, 15, dtype=torch.bool, device=device)], dim=2)

        ego_agent_features = torch.cat([ego_agent_features, torch.zeros_like(ego_agent_features[...,:1])], dim=-1)

        return ego_agent_features, ego_agent_masks, ego_agent_positions

    def encode_scene_features(self, ego_agent_features, vector_set_map_data):
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails, agent_positions = self.extract_agent_features(ego_agent_features)
        map_features, map_avails, map_positions = self.extract_map_features(vector_set_map_data, batch_size, return_positions=True)
        
        # Normalize features
        agent_features[...,:2] /= self.max_dist    # x,y
        agent_features[...,3:5] /= self.max_dist   # vx,vy
        agent_features[...,5:7] /= self.max_dist   # ax,ay
        map_features[...,:2] /= self.max_dist      # x,y

        agent_positions /= self.max_dist
        map_positions /= self.max_dist

        # Ignore distant features
        # The cutoff distance is less than the distance used for normalization
        cutoff_ratio = 1.0
        agent_avails = agent_avails * (agent_features[...,:2].norm(dim=-1) <= cutoff_ratio)
        map_avails = map_avails * (map_features[...,:2].norm(dim=-1) <= cutoff_ratio)
        agent_features[~agent_avails] = 0
        map_features[~map_avails] = 0
        
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)
        positions = torch.cat([agent_positions, map_positions], dim=1)

        # embed inputs
        feature_embedding = self.feature_embedding(features)

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = None # self.positional_embedding(features).unsqueeze(0).transpose(1, 2)

        # invalid mask
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)

        # local subgraph
        local_embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            local_embeddings = self.global_from_local(local_embeddings)
        local_embeddings = F.normalize(local_embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)

        local_map_features = local_embeddings[:,31:]
        map_pos_enc = self.rel_pos_enc(positions[:,31:])

        for layer in self.map_layers:
            local_map_features, _ = layer(local_map_features, invalid_polys[:,31:].clone(), None, None, seq1_pos=map_pos_enc)

        embeddings = torch.cat([local_embeddings[:,:31], local_map_features], dim=1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        ) # .transpose(0, 1)

        # disable certain elements on demand
        if self._feature_params.disable_agents:
            invalid_polys[
                :, 1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features))
            ] = 1  # agents won't create attention

        if self._feature_params.disable_map:
            invalid_polys[
                :, (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :
            ] = 1  # map features won't create attention

        invalid_polys[:, 0] = 0  # make ego always available in global graph

        pos_enc = self.rel_pos_enc(positions)
        for layer in self.encoder_layers:
            if self.use_positional_encodings:
                embeddings, _ = layer(embeddings, invalid_polys, None, None, seq1_pos=pos_enc, seq1_sem_pos=type_embedding)
            else:
                embeddings, _ = layer(embeddings, invalid_polys, None, None, seq1_sem_pos=type_embedding)

        return (embeddings, type_embedding, pos_enc, positions), invalid_polys

    def denoise(self, ego_trajectory, sigma, scene_features, scene_feature_masks):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        batch_size = ego_trajectory.shape[0]
        scene_features, type_embedding, scene_pos_enc, scene_pos = scene_features

        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        trajectory_time_embedding = self.trajectory_time_embeddings(torch.arange(self.H, device=ego_trajectory.device))[None].repeat(batch_size,1,1)
        trajectory_type_embedding = self.extended_type_embedding(
            torch.as_tensor([0], device=ego_trajectory.device)
        )[None].repeat(batch_size,self.H,1)
        trajectory_masks = torch.zeros(trajectory_features.shape[:-1], dtype=bool, device=trajectory_features.device)
        trajectory_pos = ego_trajectory.reshape(batch_size,16,3)[...,:2]
        trajectory_pos_enc = self.rel_pos_enc(trajectory_pos)

        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self.T
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)

        if self.use_positional_encodings:
            seq1_pos, seq2_pos = trajectory_pos_enc, scene_pos_enc
        else:
            seq1_pos, seq2_pos = None, None

        for layer in self.decoder_layers:
            trajectory_features, scene_features = layer(
                trajectory_features, trajectory_masks,
                scene_features, scene_feature_masks,
                seq1_pos=seq1_pos, seq2_pos=seq2_pos,
                seq1_sem_pos=trajectory_time_embedding,
            )

        all_features = torch.cat([scene_features, trajectory_features], dim=1)
        all_masks = torch.cat([scene_feature_masks, trajectory_masks], dim=1)
        all_type_embedding = torch.cat([type_embedding, trajectory_type_embedding], dim=1)
        all_pos_enc = torch.cat([scene_pos_enc, trajectory_pos_enc], dim=1)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        if self.use_positional_encodings:
            seq1_pos = all_pos_enc
        else:
            seq1_pos = None

        for layer in self.global_attention_layers:
            all_features, _ = layer(
                all_features, all_masks, None, None,
                seq1_pos=seq1_pos, seq1_sem_pos=all_type_embedding
            )

        trajectory_features = all_features[:,-self.H:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out
    
    def run_diffusion_es(
        self, 
        features: FeaturesType, 
        warm_start = None,
        cem_iters=20,
        temperature=0.1,
    ) -> TargetsType:
        # Recover features
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        map_features = cast(VectorSetMap, features["vector_set_map"])
        batch_size = ego_agent_features.batch_size * self.predictions_per_sample

        state_features, state_feature_masks = self.encode_scene_features(ego_agent_features, map_features)

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)

        # Multiple predictions per sample
        state_features = (
                state_features[0].repeat_interleave(self.predictions_per_sample,0),
                state_features[1].repeat_interleave(self.predictions_per_sample,0),
                state_features[2].repeat_interleave(self.predictions_per_sample,0),
                state_features[3].repeat_interleave(self.predictions_per_sample,0),
            )
        ego_agent_features = ego_agent_features.repeat_interleave(self.predictions_per_sample,0)
        state_feature_masks = state_feature_masks.repeat_interleave(self.predictions_per_sample,0)

        trunc_step_schedule = np.linspace(5,1,cem_iters).astype(int)
        noise_scale = 1.0

        trajectory_shape = (batch_size, self.H * 3)
        device = state_features[0].device

        use_warm_start = warm_start is not None

        # Initialize elite set with random 
        noise = torch.randn(trajectory_shape, device=device)
        population_trajectories, population_scores, population_info = self.rollout(
            features,
            state_features,
            state_feature_masks,
            noise,
            initial_rollout=True,
            deterministic=False,
            noise_scale=noise_scale,
        )

        # If warm start, add those to initial set
        if use_warm_start:
            prev_trajectories = warm_start
            prev_trajectories = prev_trajectories.to(device)
            # Recompute scores
            prev_scores, prev_info = compute_constraint_scores(features['constraints'], prev_trajectories)
            num_warm_samples = prev_trajectories.shape[0]
            # Concatenate to initial elite set
            population_trajectories = torch.cat([population_trajectories[:-num_warm_samples], prev_trajectories], dim=0)
            population_scores = torch.cat([population_scores[:-num_warm_samples], prev_scores], dim=0)
            population_info['traj_sim'][-num_warm_samples:] = prev_info['traj_sim']

        for i in range(cem_iters):
            population_trajectories = self.standardizer.transform_features(ego_agent_features, population_trajectories)
            n_trunc_steps = trunc_step_schedule[i]

            """
            Local MPPI update
            """
            # Compute reward-probabilities
            reward_probs = torch.exp(temperature * -population_scores)
            reward_probs = reward_probs / reward_probs.sum()
            probs = reward_probs

            """
            Resample and mutate (renoise-denoise)
            """
            indices = torch.multinomial(probs, batch_size, replacement=True) # torch.multinomial(probs, 1).squeeze(1)
            population_trajectories = population_trajectories[indices]
            population_trajectories = self.renoise(population_trajectories, n_trunc_steps)

            # Denoise
            population_trajectories, population_scores, population_info = self.rollout(
                features,
                state_features,
                state_feature_masks,
                population_trajectories,
                initial_rollout=False,
                deterministic=False,
                n_trunc_steps=n_trunc_steps,
                noise_scale=noise_scale,
            )

        if use_warm_start:
            population_trajectories = torch.cat([population_trajectories, prev_trajectories], dim=0)
            population_scores, population_info = compute_constraint_scores(features['constraints'], population_trajectories)

        best_trajectory = population_trajectories[population_scores.argmin()]
        best_trajectory = best_trajectory.reshape(-1,16,3)

        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
            "multimodal_trajectories": population_trajectories,
            "scores": population_scores,
            "traj_sim": population_info['traj_sim'][...,:2]
        }
    
    def rollout(
        self,
        features,
        state_features,
        state_feature_masks,
        ego_trajectory,
        initial_rollout=True, 
        deterministic=True, 
        n_trunc_steps=5, 
        noise_scale=1.0, 
        ablate_diffusion=False
    ):
        if initial_rollout:
            timesteps = self.scheduler.timesteps
        else:
            timesteps = self.scheduler.timesteps[-n_trunc_steps:]

        if ablate_diffusion and not initial_rollout:
            timesteps = []

        for t in timesteps:
            residual = torch.zeros_like(ego_trajectory)

            with torch.no_grad():
                residual += self.denoise(
                    ego_trajectory, 
                    t.to(ego_trajectory.device), 
                    state_features,
                    state_feature_masks
                )

            if deterministic:
                eta = 0.0
            else:
                prev_alpha = self.scheduler.alphas[t-1]
                alpha = self.scheduler.alphas[t]
                eta = noise_scale * torch.sqrt((1 - prev_alpha) / (1 - alpha)) * \
                        torch.sqrt((1 - alpha) / prev_alpha)

            out = self.scheduler.step(residual, t, ego_trajectory, eta=eta)
            ego_trajectory = out.prev_sample

        ego_agent_features = cast(AgentHistory, features["agent_history"])
        ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)
        scores, info = compute_constraint_scores(features['constraints'], ego_trajectory)

        return ego_trajectory, scores, info

    def renoise(self, ego_trajectory, t):
        noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
        ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, self.scheduler.timesteps[-t])
        return ego_trajectory


def compute_constraint_scores(constraints, trajectory):
    all_info = {}
    total_cost = torch.zeros(trajectory.shape[0], device=trajectory.device)
    for constraint in constraints:
        cost, info = constraint(trajectory)
        total_cost += cost
        all_info.update(info)
    return total_cost, all_info


def compute_constraint_residual(constraints, trajectory):
    """
    Compute the gradient of the sum of all the constraints w.r.t trajectory
    """
    with torch.enable_grad():
        trajectory.requires_grad_(True)
        total_cost = compute_constraint_scores(constraints, trajectory)
        total_cost.mean().backward()
        grad = trajectory.grad
        trajectory.requires_grad_(False)
    return grad
