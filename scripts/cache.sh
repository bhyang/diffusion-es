python nuplan-devkit/nuplan/planning/script/run_training.py \
    py_func=cache \
    cache.cache_path=/home/scratch/brianyan/test_cache/ \
    cache.force_feature_computation=True \
    +training=training_conditional_diffusion_model \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=0.01 \
    worker=single_machine_thread_pool \
    worker.use_process_pool=True
