"""lerobot-train with the last vision encoder layers of SmolVLA unfrozen.

SmolVLA's defaults (train_expert_only, freeze_vision_encoder) freeze the whole
vision-language model, so the image features cannot adapt to RoboTwin's scenes.
Unfreezing all of it does not fit in 4 GiB, so this wrapper unfreezes only the
last UNFREEZE_VISION_LAYERS SigLIP layers (default 4), the post-layernorm and the
connector, and trains them at VISION_LR_SCALE (default 0.1) times the policy's
learning rate. The rest of the VLM stays frozen. LeRobot is patched at runtime,
not edited. Arguments are passed to lerobot-train unchanged.
"""

import os

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.scripts.lerobot_train import main

NUM_LAYERS = int(os.environ.get("UNFREEZE_VISION_LAYERS", "4"))
LR_SCALE = float(os.environ.get("VISION_LR_SCALE", "0.1"))


def vision_modules(model):
    vlm = model.get_vlm_model()
    layers = vlm.vision_model.encoder.layers
    return [*layers[len(layers) - NUM_LAYERS :], vlm.vision_model.post_layernorm, vlm.connector]


original_set_requires_grad = SmolVLMWithExpertModel.set_requires_grad


def set_requires_grad(self):
    original_set_requires_grad(self)
    for module in vision_modules(self):
        for param in module.parameters():
            param.requires_grad = True


def get_optim_params(self):
    vision = {id(p) for m in vision_modules(self.model.vlm_with_expert) for p in m.parameters()}
    params = [p for p in self.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if id(p) not in vision]},
        {"params": [p for p in params if id(p) in vision], "lr": self.config.optimizer_lr * LR_SCALE},
    ]


SmolVLMWithExpertModel.set_requires_grad = set_requires_grad
SmolVLAPolicy.get_optim_params = get_optim_params

if __name__ == "__main__":
    main()
