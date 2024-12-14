import os
import textwrap
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Union
from warnings import warn
import time
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from aesthetic_scorer import AestheticScorerDiff

from accelerate.utils import ProjectConfiguration, set_seed
from transformers import is_wandb_available
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torchvision
from sd_pipeline import DiffusionPipeline
from config.rlr_config import RLR_Config
from trl.trainer import BaseTrainer, DDPOTrainer, AlignPropTrainer


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    import os
    import requests
    from tqdm import tqdm

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
    

def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn


class RLR_Trainer(BaseTrainer):
    """
    The AlignPropTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models.
    Note, this trainer is heavily inspired by the work here: https://github.com/mihirp1998/AlignProp/
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        config (`AlignPropConfig`):
            Configuration object for AlignPropTrainer. Check the documentation of `PPOConfig` for more details.
        reward_function (`Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]`):
            Reward function to be used
        prompt_function (`Callable[[], Tuple[str, Any]]`):
            Function to generate prompts to guide model
        sd_pipeline (`DiffusionPipeline`):
            Stable Diffusion pipeline to be used for training.
        image_samples_hook (`Optional[Callable[[Any, Any, Any], Any]]`):
            Hook to be called to log images
    """

    _tag_names = ["trl", "alignprop"]

    def __init__(
        self,
        config: RLR_Config,
        prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DiffusionPipeline,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.prompt_fn = prompt_function
        
        self.config = config
        self.image_samples_callback = image_samples_hook

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if "checkpoint_" not in os.path.basename(self.config.resume_from):
                # get the most recent checkpoint in this directory
                checkpoints = list(
                    filter(
                        lambda x: "checkpoint_" in x,
                        os.listdir(self.config.resume_from),
                    )
                )
                if len(checkpoints) == 0:
                    raise ValueError(f"No checkpoints found in {self.config.resume_from}")
                checkpoint_numbers = sorted([int(x.split("_")[-1]) for x in checkpoints])
                self.config.resume_from = os.path.join(
                    self.config.resume_from,
                    f"checkpoint_{checkpoint_numbers[-1]}",
                )

                accelerator_project_config.iteration = checkpoint_numbers[-1] + 1

        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        config_dict = config.to_dict()
        config_dict['checkpoints_dir'] = self.config.project_kwargs['project_dir']

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(alignprop_trainer_config=config_dict)
                if not is_using_tensorboard
                else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)

        trainable_layers = self.sd_pipeline.get_trainable_layers()

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        
        if self.config.reward_fn=='hps':
            self.loss_fn = hps_loss_fn(inference_dtype, self.accelerator.device)
        elif self.config.reward_fn=='aesthetic': # easthetic
            self.loss_fn = aesthetic_loss_fn(grad_scale=self.config.grad_scale,
                                        aesthetic_target=self.config.aesthetic_target,
                                        accelerator = self.accelerator,
                                        torch_dtype = inference_dtype,
                                        device = self.accelerator.device)
        else:
            raise NotImplementedError
        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0
        
        self.eval_prompts, self.eval_prompt_metadata = zip(*[self.prompt_fn() for _ in range(config.train_batch_size)])

    def reward_fn_RL(self, images, prompts, prompt_metadata):
        """
        Compute the reward for a given image and prompt

        Args:
            images (torch.Tensor):
                The images to compute the reward for, shape: [batch_size, 3, height, width]
            prompts (Tuple[str]):
                The prompts to compute the reward for
            prompt_metadata (Tuple[Any]):
                The metadata associated with the prompts

        Returns:
            reward (torch.Tensor), reward_metadata (Any)
        """
        if self.config.reward_fn == "hps":
            loss, rewards = self.loss_fn(images, prompts)
            return rewards, {}
        elif self.config.reward_fn == "aesthetic":
            loss, rewards = self.loss_fn(images)
            return rewards, {}
        else:
            raise NotImplementedError

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        rewards = []
        for images, prompts, prompt_metadata in prompt_image_pairs:
            reward, reward_metadata = self.reward_fn_RL(images, prompts, prompt_metadata)
            rewards.append(
                (
                    torch.as_tensor(reward, device=self.accelerator.device),
                    reward_metadata,
                )
            )
        return zip(*rewards)
    
    def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        with self.autocast():
            if self.config.train_cfg:
                noise_pred = self.sd_pipeline.unet(
                    torch.cat([latents] * 2),
                    torch.cat([timesteps] * 2),
                    embeds,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.sd_pipeline.unet(
                    latents,
                    timesteps,
                    embeds,
                ).sample
            # compute the log prob of next_latents given latents under the current model

            scheduler_step_output = self.sd_pipeline.scheduler_step(
                noise_pred,
                timesteps,
                latents,
                eta=self.config.sample_eta,
                prev_sample=next_latents,
            )

            log_prob = scheduler_step_output.log_probs

        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )

        ratio = torch.exp(log_prob - log_probs)

        loss = self.loss(advantages, self.config.train_clip_range, ratio)

        approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (List[Dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        for _i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            self.num_train_timesteps = int(self.config.sample_num_steps * self.config.timestep_fraction)
            for j in range(self.num_train_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )
                    # print(f'{j}_times')
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
        return global_step

    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.
        """
        info = defaultdict(list)
        print(f"Epoch: {epoch}, Global Step: {global_step}")

        self.sd_pipeline.unet.train()

        for _ in range(self.config.train_gradient_accumulation_steps):
            with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad():
                if self.config.gradient_estimation_strategy == "RL":
                    with torch.no_grad():
                        samples, prompt_image_pairs = self._generate_samples(
                            batch_size=self.config.train_batch_size,
                        )
                    print("Samples generated")
                    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

                    prompt_image_data = [[prompt_image_pairs["images"], prompt_image_pairs["prompts"], prompt_image_pairs["prompt_metadata"]]]
                    rewards, rewards_metadata = self.compute_rewards(prompt_image_data)

                    for i, image_data in enumerate(prompt_image_data):
                        image_data.extend([rewards[i], rewards_metadata[i]])

                    rewards = torch.cat(rewards)
                    rewards = self.accelerator.gather(rewards).detach().cpu().numpy()

                    if self.config.per_prompt_stat_tracking:
                        # gather the prompts across processes
                        prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                        prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                        advantages = self.stat_tracker.update(prompts, rewards)
                    else:
                        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                    # ungather advantages;  keep the entries corresponding to the samples on this process
                    samples["advantages"] = (
                        torch.as_tensor(advantages)
                        .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
                        .to(self.accelerator.device)
                    )

                    total_batch_size, num_timesteps = samples["timesteps"].shape

                    for inner_epoch in range(self.config.train_num_inner_epochs):
                        # shuffle samples along batch dimension
                        perm = torch.randperm(total_batch_size, device=self.accelerator.device)
                        samples = {k: v[perm] for k, v in samples.items()}

                        # shuffle along time dimension independently for each sample
                        # still trying to understand the code below
                        perms = torch.stack(
                            [torch.randperm(num_timesteps, device=self.accelerator.device) for _ in range(total_batch_size)]
                        )

                        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                            samples[key] = samples[key][
                                torch.arange(total_batch_size, device=self.accelerator.device)[:, None],
                                perms,
                            ]

                        original_keys = samples.keys()
                        original_values = samples.values()
                        # rebatch them as user defined train_batch_size is different from sample_batch_size
                        reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

                        # Transpose the list of original values
                        transposed_values = zip(*reshaped_values)
                        # Create new dictionaries for each row of transposed values
                        samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]

                        # self.sd_pipeline.unet.train()
                        global_step = self._train_batched_samples(inner_epoch, epoch, global_step, samples_batched)
                        # ensure optimization step at the end of the inner epoch
                        if not self.accelerator.sync_gradients:
                            raise ValueError(
                                "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
                            )


                else:
                    samples, prompt_image_pairs = self._generate_samples(
                            batch_size=self.config.train_batch_size,
                        )
                    print("Samples generated")
                    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

                    if "hps" in self.config.reward_fn:
                        loss, rewards = self.loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                    else:
                        loss, rewards = self.loss_fn(prompt_image_pairs["images"])

                    rewards = self.accelerator.gather(rewards).detach().cpu().numpy()
                    loss = loss.mean()
                    loss = loss * self.config.loss_coeff
                    
                    # if self.config.gradient_estimation_strategy != "LR":
                    #     self.accelerator.backward(loss)
                    # else:
                    #     loss = loss.detach()
                    #     log_probs = samples["log_probs"]
                    #     loss_target = torch.mean(loss*log_probs.sum())
                    #     self.accelerator.backward(loss_target)

                    if self.config.gradient_estimation_strategy != "LR":
                        loss = loss.mean()
                        loss = loss * self.config.loss_coeff
                        self.accelerator.backward(loss)
                    else:
                        loss = loss.detach()
                        log_probs = torch.sum(samples["log_probs"], dim=1)
                        loss_target = torch.mean(loss*log_probs) * self.config.loss_coeff
                        self.accelerator.backward(loss_target)
                        loss = loss.mean()

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    info["reward_mean"].append(rewards.mean())
                    info["reward_std"].append(rewards.std())
                    info["loss"].append(loss.item())

        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
            info.update({"epoch": epoch})
            self.accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)
        else:
            raise ValueError(
                "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
            )
        # Logs generated images
        if self.image_samples_callback is not None and global_step % self.config.log_image_freq == 0 and self.accelerator.is_main_process:
            print("Logging images")
            # Fix the random seed for reproducibility
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            _, prompt_image_pairs = self._generate_samples(
                    batch_size=self.config.train_batch_size, with_grad=False, prompts=self.eval_prompts
                )
            self.image_samples_callback(prompt_image_pairs, global_step, self.accelerator.trackers[0])
            seed = random.randint(0, 100)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)            

        if epoch != 0 and epoch % self.config.save_freq == 0:
            print("Saving checkpoint")
            self.accelerator.save_state()
        print("Step Done")
        return global_step

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def _generate_samples(self, batch_size, with_grad=True, prompts=None):
        """
        Generate samples from the model

        Args:
            batch_size (int): Batch size to use for sampling
            with_grad (bool): Whether the generated RGBs should have gradients attached to it.

        Returns:
            
        """
        # original code
        # samples = {}

        samples = []
        # prompt_image_pairs = []
        prompt_image_pairs = {}

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        if prompts is None:
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])
        else:
            prompt_metadata = [{} for _ in range(batch_size)]

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)

        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

        if with_grad:
            sd_output = self.sd_pipeline.rgb_with_grad(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                backprop_strategy=self.config.gradient_estimation_strategy,
                backprop_kwargs=self.config.backprop_kwargs[self.config.gradient_estimation_strategy],
                output_type="pt",
            )
        else:
            sd_output = self.sd_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                output_type="pt",
            )

        images = sd_output.images
        latents = sd_output.latents
        log_probs = sd_output.log_probs

        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
        timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],  # each entry is the latent before timestep t
                "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "negative_prompt_embeds": sample_neg_prompt_embeds,
            }
        )
        # prompt_image_pairs.append([images, prompts, prompt_metadata])
        prompt_image_pairs["images"] = images
        prompt_image_pairs["prompts"] = prompts
        prompt_image_pairs["prompt_metadata"] = prompt_metadata

        return samples, prompt_image_pairs

        # orignial code
        # images = sd_output.images
        # samples["images"] = images
        # samples["prompts"] = prompts
        # samples["prompt_metadata"] = prompt_metadata
        # samples["all_log_probs"] = sd_output.log_probs

        # return samples
        

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        if epochs is None:
            epochs = self.config.num_epochs
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)

    def _save_pretrained(self, save_directory):
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card()
