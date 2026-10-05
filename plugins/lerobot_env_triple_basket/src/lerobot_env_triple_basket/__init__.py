"""LeRobot plugin: registers the LIBERO suite ``libero_triple_basket``.

LeRobot imports every installed distribution named ``lerobot_env_*`` at start-up
(``register_third_party_plugins``), so after ``pip install -e plugins/lerobot_env_triple_basket``
this works with the unmodified vendor checkout::

    lerobot-eval --env.type=libero --env.task=libero_triple_basket --env.task_ids='[0]' ...

Task 0: "put all three items in the basket" (item_1, item_2, item_3 are identical ketchup
bottles at three fixed positions). Success = all three are inside the basket's contain
region AND stay there for ``SUCCESS_DWELL_STEPS`` consecutive checks.
"""
from pathlib import Path

SUITE_NAME = "libero_triple_basket"
TASK_NAME = "put_all_three_items_in_the_basket"
ITEMS = ("item_1", "item_2", "item_3")
BASKET = "basket_1"
SUCCESS_DWELL_STEPS = 5  # consecutive env steps all three must stay inside the basket

DATA_DIR = Path(__file__).parent / "data"
BDDL_DIR = DATA_DIR / "bddl"
INIT_DIR = DATA_DIR / "init_states"
BDDL_FILE = BDDL_DIR / f"{TASK_NAME}.bddl"
INIT_FILE = INIT_DIR / f"{TASK_NAME}.pruned_init"


def _register() -> None:
    from libero.libero.benchmark import Benchmark, Task, register_benchmark
    from libero.libero.envs import env_wrapper

    # `problem_folder` is an absolute path: both LeRobot (os.path.join / pathlib "/")
    # and LIBERO drop the LIBERO root when the second component is absolute.
    task = Task(
        name=TASK_NAME,
        language="put all three items in the basket",
        problem="Libero",
        problem_folder=str(BDDL_DIR),
        bddl_file=BDDL_FILE.name,
        init_states_file=INIT_FILE.name,
    )

    @register_benchmark
    class LIBERO_TRIPLE_BASKET(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = SUITE_NAME
            self.tasks = [task]
            self.n_tasks = 1

        def get_task_init_states(self, i):
            import torch

            return torch.load(INIT_DIR / self.tasks[i].init_states_file, weights_only=False)

    # LeRobot reads `info["is_success"]` from env.check_success(). Add a dwell requirement
    # (all three must stay inside for N steps) for this task only; LIBERO's native
    # goal is already the AND over the three `In` predicates.
    if getattr(env_wrapper.ControlEnv, "_triple_basket_patched", False):
        return
    original = env_wrapper.ControlEnv.check_success

    def check_success(self):
        raw = original(self)
        if TASK_NAME not in str(getattr(self.env, "bddl_file_name", "")):
            return raw
        self._dwell = (getattr(self, "_dwell", 0) + 1) if raw else 0
        return self._dwell >= SUCCESS_DWELL_STEPS

    env_wrapper.ControlEnv.check_success = check_success
    env_wrapper.ControlEnv._triple_basket_patched = True


try:
    _register()
except ImportError:  # libero not installed: plugin stays inert
    pass
