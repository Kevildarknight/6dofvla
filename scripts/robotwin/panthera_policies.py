"""Runtime patches that let LeRobot's Diffusion Policy take the Panthera cameras.

LeRobot's DiffusionPolicy stacks all cameras into one tensor, so it requires
every image feature to have the same shape, while RoboTwin's head camera is
240x320 and the wrist camera 360x640. With --policy.resize_shape=[H,W], these
patches declare every camera as H x W and resize each image to that size before
the policy sees it (bilinear, antialiased), in training and at inference.
LeRobot is patched at runtime, not edited; import this module before building
or loading a diffusion policy.
"""

import torch.nn.functional as F  # noqa: N812
from lerobot.configs.types import PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

original_validate_features = DiffusionConfig.validate_features


def validate_features(self):
    if self.resize_shape is not None:
        for key, feature in self.image_features.items():
            self.input_features[key] = PolicyFeature(feature.type, (feature.shape[0], *self.resize_shape))
    original_validate_features(self)


def resize_images(config, batch):
    if config.resize_shape is None:
        return batch
    size = tuple(config.resize_shape)
    for key in config.image_features:
        image = batch.get(key)
        if image is None or tuple(image.shape[-2:]) == size:
            continue
        lead = image.shape[:-3]
        image = F.interpolate(image.reshape(-1, *image.shape[-3:]), size=size, mode="bilinear", antialias=True)
        batch[key] = image.reshape(*lead, *image.shape[-3:])
    return batch


def resizing(method):
    def wrapper(self, batch, *args, **kwargs):
        return method(self, resize_images(self.config, dict(batch)), *args, **kwargs)

    return wrapper


DiffusionConfig.validate_features = validate_features
for name in ("forward", "predict_action_chunk", "select_action"):
    setattr(DiffusionPolicy, name, resizing(getattr(DiffusionPolicy, name)))
