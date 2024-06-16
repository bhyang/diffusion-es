python nuplan-devkit/nuplan/planning/script/run_training.py \
    experiment_name=kinematic \
    job_name=kinematic \
    py_func=train \
    +training=training_kinematic_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/test_cache \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=16 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint_callback.every_n_epochs=10 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    callbacks.visualization_callback.skip_train=False \
    model.T=100
    
