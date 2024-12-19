CUDA_VISIBLE_DEVICES=2 accelerate launch train_with_trl.py \
    --num_epochs=200 \
    --sample_num_steps=50 \
    --sample_num_batches_per_epoch=4 \
    --sample_batch_size=8 \
    --train_batch_size=4 \
    --train_gradient_accumulation_steps=2 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"