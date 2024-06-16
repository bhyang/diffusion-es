
SPLIT=reduced_val14_split # val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
model=kinematic_diffusion_model \
planner=pdm_diffusion_planner \
planner.pdm_diffusion_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/conditional-refactor/conditional/2023.10.19.16.54.21/checkpoints/epoch\=490.ckpt" \
planner.pdm_diffusion_planner.dump_gifs_path="/home/scratch/brianyan/viz/eval_cond/" \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
number_of_gpus_allocated_per_simulation=0.1 \
experiment_name=eval_cond \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
worker.threads_per_node=40
