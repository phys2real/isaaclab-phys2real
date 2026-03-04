# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

def save_code_snapshot(snapshot_path: str, resume_path: str | None, args_cli, hydra_args):
    """
    Write a single CODE_SNAPSHOT file with:
      - timestamp, cwd
      - HEAD commit, branch, git describe
      - CLI and Hydra args
      - git status (porcelain)
      - changed + untracked file lists
      - DIFF (staged) and DIFF (unstaged), both relative to HEAD
    """
    import subprocess, os, textwrap, datetime, json

    def _run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "replace").strip()
        except Exception as e:
            return f"<error running {' '.join(cmd)}: {e}>"

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = []

    lines.append(f"# CODE SNAPSHOT\n")
    lines.append(f"timestamp: {ts}")
    lines.append(f"cwd: {os.getcwd()}")
    lines.append(f"resume_checkpoint: {resume_path if resume_path else '<none>'}")

    if _run(["git", "rev-parse", "--is-inside-work-tree"]) != "true":
        lines.append("\n<not a git repository — no VCS details available>\n")
        content = "\n".join(lines) + "\n"
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        with open(snapshot_path, "w") as f:
            f.write(content)
        return

    head     = _run(["git", "rev-parse", "HEAD"])
    branch   = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    describe = _run(["git", "describe", "--tags", "--always", "--dirty"])
    repo_top = _run(["git", "rev-parse", "--show-toplevel"])
    status   = _run(["git", "status", "--porcelain=v1"])
    showstat = _run(["git", "show", "--stat", "--oneline", "-s", head])

    # derive file lists from porcelain
    changed_files = []
    untracked_files = []
    for line in status.splitlines():
        if line.startswith("?? "):
            untracked_files.append(line[3:])
        elif len(line) >= 4:
            changed_files.append(line[3:])

    diff_staged   = _run(["git", "diff", "--cached"])
    diff_unstaged = _run(["git", "diff"])
    
    # Also capture rl_games git diff
    rl_games_path = "/home/maggiewang/Workspace/rl_games_isaacsim"
    def _run_in_dir(cmd, cwd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd).decode("utf-8", "replace").strip()
        except Exception as e:
            return f"<error running {' '.join(cmd)} in {cwd}: {e}>"
    
    rl_games_diff_staged = ""
    rl_games_diff_unstaged = ""
    rl_games_head = ""
    rl_games_branch = ""
    if os.path.exists(rl_games_path):
        rl_games_head = _run_in_dir(["git", "rev-parse", "HEAD"], rl_games_path)
        rl_games_branch = _run_in_dir(["git", "rev-parse", "--abbrev-ref", "HEAD"], rl_games_path)
        rl_games_diff_staged = _run_in_dir(["git", "diff", "--cached"], rl_games_path)
        rl_games_diff_unstaged = _run_in_dir(["git", "diff"], rl_games_path)

    # Header
    lines.append(f"repo_root: {repo_top}")
    lines.append(f"branch: {branch}")
    lines.append(f"HEAD: {head}")
    lines.append(f"describe: {describe}")
    lines.append(f"head_oneline: {showstat}")

    # Args
    lines.append("\n## ARGS")
    lines.append("CLI:\n" + json.dumps(vars(args_cli), indent=2, default=str))
    lines.append("Hydra (raw):\n" + " ".join(hydra_args))

    # Status
    lines.append("\n## GIT STATUS (porcelain=v1)")
    lines.append(status if status else "<clean>")

    lines.append("\n## CHANGED FILES (tracked)")
    lines.append("\n".join(changed_files) if changed_files else "<none>")

    lines.append("\n## UNTRACKED FILES")
    lines.append("\n".join(untracked_files) if untracked_files else "<none>")

    # Diffs
    lines.append("\n## DIFF (staged vs HEAD)")
    lines.append(diff_staged if diff_staged else "<none>")

    lines.append("\n## DIFF (unstaged vs HEAD)")
    lines.append(diff_unstaged if diff_unstaged else "<none>")
    
    # rl_games repository info
    lines.append(f"\n## RL_GAMES REPOSITORY ({rl_games_path})")
    if os.path.exists(rl_games_path):
        lines.append(f"rl_games_branch: {rl_games_branch}")
        lines.append(f"rl_games_HEAD: {rl_games_head}")
        lines.append("\n## RL_GAMES DIFF (staged vs HEAD)")
        lines.append(rl_games_diff_staged if rl_games_diff_staged else "<none>")
        lines.append("\n## RL_GAMES DIFF (unstaged vs HEAD)")
        lines.append(rl_games_diff_unstaged if rl_games_diff_unstaged else "<none>")
    else:
        lines.append("<rl_games path does not exist>")

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    with open(snapshot_path, "w") as f:
        f.write(content)

@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    # --- snapshot: only on global rank 0 ---
    global_rank_for_snapshot = int(os.getenv("RANK", "0"))
    if global_rank_for_snapshot == 0:
        resume_path_for_note = resume_path if args_cli.checkpoint is not None else None
        snapshot_path = os.path.join(log_root_path, log_dir, "CODE_SNAPSHOT.txt")
        save_code_snapshot(snapshot_path, resume_path_for_note, args_cli, hydra_args)
    # --- end snapshot ---

    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"env_cfg": env_cfg.to_dict()})
        wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
