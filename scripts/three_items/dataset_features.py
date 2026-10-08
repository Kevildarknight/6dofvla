"""Dataset layout shared by the demo collector and the LIBERO subset converter.

Feature names and shapes match HuggingFaceVLA/libero and the inputs of
models/smolvla_libero. Images are stored as video to save space.

FPS is 10 to match HuggingFaceVLA/libero, so the two datasets can be merged.
In both, one frame is one 20 Hz control step; the label only sets video playback
speed and the spacing of timestamps, and SmolVLA's action chunks are indexed by
frame, so the label does not change what the policy learns.
"""

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
SIZE = 256
FPS = 10

FEATURES = {
    **{
        key: {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channels"]}
        for key in IMAGE_KEYS
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper_1", "gripper_2"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}
