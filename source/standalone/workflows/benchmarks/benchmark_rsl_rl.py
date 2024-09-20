# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from omni.isaac.lab.app import AppLauncher

import source.standalone.workflows.rsl_rl.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="OsmoKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile"],
    help="Benchmarking backend options, defaults OsmoKPIFile",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# to ensure kit args don't break the benchmark arg parsing
args_cli, _ = parser.parse_known_args()

app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

app_start_time_end = time.perf_counter_ns()

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

imports_time_end = time.perf_counter_ns()

from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.benchmark.services")
from omni.isaac.benchmark.services import BaseIsaacBenchmark

from omni.isaac.lab.utils.timer import Timer
from source.standalone.workflows.benchmarks.utils import (
    log_app_start_time,
    log_python_imports_time,
    log_rl_policy_episode_lengths,
    log_rl_policy_rewards,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_task_start_time,
    log_total_start_time,
    parse_tf_logs,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Create the benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_rsl_rl_train",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "max_iterations", "data": args_cli.max_iterations},
        ]
    },
    backend_type=args_cli.benchmark_backend,
)


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    benchmark.set_phase("loading", start_recording_frametime=False, start_recording_runtime=True)
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    task_startup_time_end = time.perf_counter_ns()

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    benchmark.set_phase("sim_runtime")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    benchmark.store_measurements()

    # parse tensorboard file stats
    log_data = parse_tf_logs(log_dir)

    # prepare RL timing dict
    rl_training_times = {
        "Collection Time": log_data["Perf/collection time"],
        "Learning Time": log_data["Perf/learning_time"],
        "Total FPS": log_data["Perf/total_fps"],
    }

    # log additional metrics to benchmark services
    log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
    log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
    log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)
    log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
    log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
    log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
    log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
    log_rl_policy_rewards(benchmark, log_data["Train/mean_reward"])
    log_rl_policy_episode_lengths(benchmark, log_data["Train/mean_episode_length"])

    benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
