# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .string_target import ID_TO_TARGET, NUM_TARGETS, PREDEFINED_TARGETS, TARGET_TO_ID  # 假设定义在一个单独的文件

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.manager_based.manipulation.my_lift.lift_env import LiftEnv


def randomize_string_task_goal(env: LiftEnv, env_ids: torch.Tensor):  # 类型提示为你的环境类
    """
    Randomizes the string-based task goal for the specified environment instances.
    Updates the environment's internal state for the current target ID,
    its one-hot encoding, and the actual string representation.

    It also (optionally, if implemented) triggers scene updates based on the new goal.
    """

    if env_ids is None:
        # "startup" 模式或者其他需要作用于所有环境的情况
        effective_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        num_to_resample = env.num_envs
    else:
        # "reset" 模式，env_ids 是一个包含需要重置的环境索引的张量
        effective_env_ids = env_ids
        num_to_resample = len(env_ids)  # 现在 env_ids 不是 None 了
    if num_to_resample == 0:
        return

    # 1. 随机采样新的目标 ID
    sampled_indices = torch.randint(0, NUM_TARGETS, (num_to_resample,), device=env.device)

    # 2. 更新环境实例中存储的目标信息
    #    确保你的 MyLiftEnv 类有这些属性，并且在 __init__ 中正确初始化了它们。
    if (
        not hasattr(env, "current_target_ids_per_env")
        or not hasattr(env, "encoded_task_goal_per_env")
        or not hasattr(env, "current_target_strings_per_env")
    ):
        raise AttributeError(
            "The environment object is missing required target attributes "
            " (e.g., current_target_ids_per_env). "
            "Ensure they are initialized in the env's __init__."
        )

    env.current_target_ids_per_env[effective_env_ids] = sampled_indices
    # env.encoded_task_goal_per_env[effective_env_ids] = torch.nn.functional.one_hot(
    #     sampled_indices, num_classes=NUM_TARGETS
    # ).float()

    for i, env_idx in enumerate(effective_env_ids.tolist()):
        env.current_target_strings_per_env[env_idx] = ID_TO_TARGET[sampled_indices[i].item()]
        # 可选：调试打印
        if hasattr(env.cfg, "print_debug_info") and env.cfg.print_debug_info:
            print(f"Env {env_idx} new target (via event): {env.current_target_strings_per_env[env_idx]}")

    inputs = env.text_processor(
        text=env.current_target_strings_per_env, padding="max_length", max_length=64, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        outputs = env.siglip_text_model(**inputs)

    env.current_target_state_per_env = outputs.pooler_output  # pooled features


def reset_database(env: LiftEnv, env_ids: torch.Tensor):
    """
    Deletes all items from the ChromaDB collection that belong to the specified environment IDs.

    This is typically called in the environment's _reset_idx method.
    """
    # 检查是否有需要重置的环境
    if env_ids.numel() == 0:
        return

    # 将 env_ids 转换为 Python 的 list，因为 ChromaDB 的 $in 操作符需要它
    env_ids_list = env_ids.cpu().tolist()

    # 直接使用 where 子句来删除所有匹配的条目
    # 这只需要一次数据库交互，更高效
    env.collection.delete(where={"env_id": {"$in": env_ids_list}})

    # 你可以加上日志来确认操作
    # print(f"ChromaDB: Deleted all items for env_ids: {env_ids_list}")
