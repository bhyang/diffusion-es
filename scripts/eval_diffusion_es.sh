
SPLIT=val14_split # val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
model=kinematic_diffusion_model \
planner=pdm_diffusion_planner \
planner.pdm_diffusion_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/kinematic_v2/no_hist/2023.10.13.11.46.37/best_model/epoch\=490.ckpt" \
planner.pdm_diffusion_planner.dump_gifs_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/viz/eval_diffusion_full_ttc/" \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
number_of_gpus_allocated_per_simulation=0.125 \
experiment_name=repo_sanity_check \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
planner.pdm_diffusion_planner.follow_centerline=True \
planner.pdm_diffusion_planner.scorer_config.ttc_fixed_speed=False \
ego_controller=one_stage_controller \
worker.threads_per_node=32
