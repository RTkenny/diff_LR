CUDA_VISIBLE_DEVICES=1 accelerate launch main.py \
    --num_epochs=200 \
    --gradient_estimation_strategy='RL' \
    --sample_num_steps=50 \
    --sample_eta=1.0 \
    --sample_num_batches_per_epoch=4 \
    --sample_batch_size=8 \
    --train_num_inner_epochs=1 \
    --train_batch_size=4 \
    --train_gradient_accumulation_steps=2 \
    --reward_fn='aesthetic' \
    --prompt_fn='simple_animals' \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"