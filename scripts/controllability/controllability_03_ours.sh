
SPLIT=controllability_03_unprotected_right # val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
EXPERIMENT_NAME=controllability_03_ours
RESULT_PATH=/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/viz/results_$EXPERIMENT_NAME.json

for SEED in {1..10}
do
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    model=kinematic_diffusion_model \
    planner=pdm_diffusion_language_planner \
    planner.pdm_diffusion_language_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/kinematic_v2/no_hist/2023.10.13.11.46.37/best_model/epoch\=490.ckpt" \
    planner.pdm_diffusion_language_planner.dump_gifs_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/viz/" \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    number_of_gpus_allocated_per_simulation=1.0 \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    worker=sequential \
    observation=noisy_idm_agents_observation \
    metric_aggregator=closed_loop_reactive_agents_weighted_average_no_prog \
    ego_controller=one_stage_controller \
    seed=$SEED \
    experiment_name=$EXPERIMENT_NAME \
    planner.pdm_diffusion_language_planner.experiment_log_path=$RESULT_PATH \
    planner.pdm_diffusion_language_planner.nuplan_output_dir=\${output_dir} \
    planner.pdm_diffusion_language_planner.language_config.instruction="Change to lane 33."
done

python process_results.py --result_path $RESULT_PATH
