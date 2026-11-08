"""Run RoboTwin's script/eval_policy.py with the training ("seen") background textures.

eval_policy.py always sets eval_mode, and with random_background on, eval_mode
makes Base_Task.create_table_and_wall draw table and wall textures from
assets/background_texture/unseen instead of seen. That makes every evaluation a
test on new textures. This wrapper turns eval_mode off only while the table and
wall are created, so evaluation uses the same texture pool as data collection.
eval_mode's other effect, the per-task step limit, is kept.

Run from RoboTwin's root folder with eval_policy.py's own arguments; eval.sh
does this when TEXTURES=seen.
"""

import runpy
import sys

sys.path[:0] = ["./script", "./"]  # as if run as script/eval_policy.py from RoboTwin's root
from envs._base_task import Base_Task  # noqa: E402

create_table_and_wall = Base_Task.create_table_and_wall


def create_table_and_wall_seen(self, *args, **kwargs):
    eval_mode, self.eval_mode = self.eval_mode, False
    try:
        return create_table_and_wall(self, *args, **kwargs)
    finally:
        self.eval_mode = eval_mode


Base_Task.create_table_and_wall = create_table_and_wall_seen
sys.argv[0] = "script/eval_policy.py"
runpy.run_path("script/eval_policy.py", run_name="__main__")
