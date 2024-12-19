import ipdb
st = ipdb.set_trace
import builtins
import time
import os
builtins.st = ipdb.set_trace
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser, is_wandb_available

from config.rlr_config import RLR_Config
from rlr_trainer import RLR_Trainer
from sd_pipeline import DiffusionPipeline
from trl.models.auxiliary_modules import aesthetic_scorer
import tempfile
from PIL import Image
if is_wandb_available():
    import wandb

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="/home/rt/data/SD_playground/CompVis/stable-diffusion-v1-4", metadata={"help": "the pretrained model to use"} # "runwayml/stable-diffusion-v1-5" original /root/autodl-tmp/CompVis/stable-diffusion-v1-4
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})




def image_outputs_logger(image_data, global_step, accelerate_logger):

    with tempfile.TemporaryDirectory() as tmpdir:
                images = image_data[0][0]
                prompts = image_data[0][1]
                for i, image in enumerate(images):
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                accelerate_logger.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{i}.jpg"),
                                caption=f"{prompt:.25}",
                            )
                            for i, prompt in enumerate(
                                prompts
                            )  # only log rewards from process 0
                        ],
                    },
                    step=global_step,
                )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RLR_Config))
    script_args, training_args = parser.parse_args_into_dataclasses()
    project_dir = f"alignprop_{int(time.time())}"
    os.makedirs(f"checkpoints/{project_dir}", exist_ok=True)
    
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"checkpoints/{project_dir}",
    }

    prompt_fn = getattr(prompts_file, training_args.prompt_fn)
    
    pipeline = DiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    trainer = RLR_Trainer(
        training_args,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()