"""lerobot-train with panthera_policies.py applied, for Diffusion Policy on RoboTwin Panthera data.

Arguments are passed to lerobot-train unchanged; see train_diffusion.sh.
"""

import panthera_policies  # noqa: F401
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
