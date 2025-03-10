CUDA_VISIBLE_DEVICES=0 accelerate launch ./scripts/main.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=2 \
    --gradient_estimation_strategy='LR' \
    --sample_num_steps=50 \
    --reward_fn='aesthetic' \
    --prompt_fn='simple_animals' \
    --train_batch_size=2 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"