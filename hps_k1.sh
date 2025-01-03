# i have tested the code with 128 batch size, i.e 4 gpus x 8 batch size x 4 gradient accumulation steps, however you can change the batch size 
# or batch size division as per your requirements
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch main.py \
    --num_epochs=20 \
    --train_gradient_accumulation_steps=8 \
    --backprop_strategy='fixed' \
    --sample_num_steps=50 \
    --reward_fn='hps' \
    --prompt_fn='hps_v2_all' \
    --train_batch_size=4 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"