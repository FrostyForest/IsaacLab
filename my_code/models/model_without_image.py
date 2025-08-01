from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
import torch.nn as nn
from models import utils


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        init_log_std=0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.state_net1 = nn.Sequential(
            nn.Linear(26, 128),  # ik rel input 37，原38
            # nn.BatchNorm1d(128),
            nn.ELU(),
            # nn.Linear(64, 16),
            # nn.ELU(),
        )
        self.state_net2 = nn.Sequential(
            nn.Linear(15, 128),  # ik rel input 37，原38
            # nn.BatchNorm1d(128),
            nn.ELU(),
            # nn.Linear(64, 16),
            # nn.ELU(),
        )
        # self.rgb_net = utils.CustomFeatureExtractor(in_channels=16, output_dim=16)
        # self.net = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 128),nn.LayerNorm(128),nn.ELU(),nn.Linear(128, 64), nn.ELU())
        self.block1 = utils.ResidualBlock(feature_size=256)
        self.block2 = utils.ResidualBlock(feature_size=256)
        self.action_block = utils.ResidualBlock(feature_size=256)

        self.mean_layer = nn.Sequential(
            nn.Linear(256, self.num_actions),
            # nn.ELU(),
            # nn.Linear(self.num_actions, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions) * init_log_std)

        self.value_layer = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.bool))

        # self.register_buffer('joint_pos_mean', torch.zeros(1))
        # self.register_buffer('joint_pos_std', torch.ones(1))
        # self.register_buffer('joint_vel_mean', torch.zeros(1))
        # self.register_buffer('joint_vel_std', torch.ones(1))
        # self.register_buffer('action_mean', torch.zeros(1))
        # self.register_buffer('action_std', torch.ones(1))
        # self.register_buffer('object_position_mean', torch.zeros(1))
        # self.register_buffer('object_position_std', torch.ones(1))
        # self.register_buffer('ee_position_mean', torch.zeros(1))
        # self.register_buffer('ee_position_std', torch.ones(1))
        # self.register_buffer('target_object_position_mean', torch.zeros(1))
        # self.register_buffer('target_object_position_std', torch.ones(1))
        # 定义一个字典，包含我们想要进行归一化的观测键及其对应的空间
        # 对于'actions'，我们从 action_space 获取其空间
        self.normalization_spaces = {
            "joint_pos": observation_space["joint_pos"],
            "joint_vel": observation_space["joint_vel"],
            "object_position": observation_space["object_position"],
            "ee_position": observation_space["ee_position"],
            "target_object_position": observation_space["target_object_position"],
            "actions": observation_space["actions"],  # 你的 space 中包含了过去的 actions
        }

        # 动态地注册具有正确维度的缓冲区
        for key, space in self.normalization_spaces.items():
            # 获取维度，通常是 space.shape 元组的第一个元素
            if not hasattr(space, "shape") or not len(space.shape):
                raise ValueError(f"空间 '{key}' 没有有效的 shape 属性")

            dim = space.shape[0]

            # 为这个键注册均值和标准差的缓冲区
            self.register_buffer(f"{key}_mean", torch.zeros(dim, device=device))
            self.register_buffer(f"{key}_std", torch.ones(dim, device=device))
            print(f"为 '{key}' 注册了维度为 {dim} 的归一化缓冲区。")

        self.ema_momentum = 0.975
        self._shared_output = None

    def _initialization_hook(self, module, inputs):
        """这个钩子函数会在每次 forward 调用前执行"""
        # inputs 是一个包含所有 forward 输入的元组
        space = self.tensor_to_space(inputs["states"], self.observation_space)

        # 检查标志位，并确保只在训练时初始化
        if module.training:
            print("Hook triggered: Initializing parameter.")
            with torch.no_grad():
                # --- 优化: 将所有更新操作放在一个 no_grad() 上下文中 ---
                # --- 修复和优化: 直接用 .copy_() 更新 buffer 的值 ---
                momentum = self.ema_momentum
                # 遍历所有键，并更新对应的缓冲区
                for key in self.normalization_spaces.keys():
                    mean_buffer_name = f"{key}_mean"
                    std_buffer_name = f"{key}_std"

                    # 获取当前批次对应键的数据
                    current_data = space[key]

                    # ==================== 关键改动 ====================
                    # 沿着批次维度（dim=0）计算均值和标准差
                    current_mean = current_data.mean(dim=0)
                    current_std = current_data.std(dim=0)
                    # ===============================================

                    # 获取旧的缓冲区张量
                    old_mean = getattr(module, mean_buffer_name)
                    old_std = getattr(module, std_buffer_name)

                    # 指数移动平均更新
                    new_mean = old_mean * momentum + current_mean * (1 - momentum)
                    new_std = old_std * momentum + current_std * (1 - momentum)

                    # 将新值复制回缓冲区
                    old_mean.copy_(new_mean)
                    old_std.copy_(new_std.clamp(min=1e-6))
                # # --- joint_pos ---
                # current_mean = space["joint_pos"].mean()
                # new_mean = module.joint_pos_mean * momentum + current_mean * (1 - momentum)
                # module.joint_pos_mean.copy_(new_mean)

                # current_std = space["joint_pos"].std()
                # new_std = module.joint_pos_std * momentum + current_std * (1 - momentum)
                # module.joint_pos_std.copy_(new_std.clamp(min=1e-6))

                # # --- joint_vel ---
                # current_mean = space["joint_vel"].mean()
                # new_mean = module.joint_vel_mean * momentum + current_mean * (1 - momentum)
                # module.joint_vel_mean.copy_(new_mean)

                # current_std = space["joint_vel"].std()
                # new_std = module.joint_vel_std * momentum + current_std * (1 - momentum)
                # module.joint_vel_std.copy_(new_std.clamp(min=1e-6))

                # # --- actions ---
                # current_mean = space["actions"].mean()
                # new_mean = module.action_mean * momentum + current_mean * (1 - momentum)
                # module.action_mean.copy_(new_mean)

                # current_std = space["actions"].std()
                # new_std = module.action_std * momentum + current_std * (1 - momentum)
                # module.action_std.copy_(new_std.clamp(min=1e-6))

                # # --- object_position ---
                # current_mean = space["object_position"].mean()
                # new_mean = module.object_position_mean * momentum + current_mean * (1 - momentum)
                # module.object_position_mean.copy_(new_mean)

                # current_std = space["object_position"].std()
                # new_std = module.object_position_std * momentum + current_std * (1 - momentum)
                # module.object_position_std.copy_(new_std.clamp(min=1e-6))

                # # --- ee_position ---
                # current_mean = space["ee_position"].mean()
                # new_mean = module.ee_position_mean * momentum + current_mean * (1 - momentum)
                # module.ee_position_mean.copy_(new_mean)

                # current_std = space["ee_position"].std()
                # new_std = module.ee_position_std * momentum + current_std * (1 - momentum)
                # module.ee_position_std.copy_(new_std.clamp(min=1e-6))

                # # --- target_object_position ---
                # current_mean = space["target_object_position"].mean()
                # new_mean = module.target_object_position_mean * momentum + current_mean * (1 - momentum)
                # module.target_object_position_mean.copy_(new_mean)

                # current_std = space["target_object_position"].std()
                # new_std = module.target_object_position_std * momentum + current_std * (1 - momentum)
                # module.target_object_position_std.copy_(new_std.clamp(min=1e-6))

            # 更新标志位
            module.initialized.fill_(True)

            print(f"Parameter initialized to: {module.data_param.data}")

            # (可选) 初始化完成后，可以移除钩子以提高性能
            # self._init_hook_handle.remove()

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):

        space = self.tensor_to_space(inputs["states"], self.observation_space)
        if role == "policy":

            state_data1 = (
                torch.cat(
                    [
                        (space["joint_pos"] - self.joint_pos_mean) / self.joint_pos_std,
                        (space["joint_vel"] - self.joint_vel_mean) / self.joint_vel_std,
                        # space["ee_position"],
                        # space["target_object_position"],
                        (space["actions"] - self.actions_mean) / self.actions_std,
                        # space["contact_force_left_finger"],
                        # space["contact_force_left_finger"],
                    ],
                    dim=-1,
                )
                .to(torch.float32)
                .clamp(max=10, min=-10)
            )
            state_data2 = (
                torch.cat(
                    [
                        # space["joint_pos"],
                        # space["joint_vel"],
                        (space["object_position"] - self.object_position_mean) / self.object_position_std,  # 3
                        (space["ee_position"] - self.ee_position_mean) / self.ee_position_std,  # 3
                        (space["target_object_position"] - self.target_object_position_mean)
                        / self.target_object_position_std,  # 7
                        # space["actions"],
                        space["contact_force_left_finger"],  # 1
                        space["contact_force_right_finger"],  # 1
                    ],
                    dim=-1,
                )
                .to(torch.float32)
                .clamp(max=10, min=-10)
            )

            feature1 = self.state_net1(state_data1)
            feature3 = self.state_net2(state_data2)

            r1 = self.block1(torch.cat([feature1, feature3], dim=-1))
            r2 = self.block2(r1)
            self._shared_output = r2
            r3 = self.action_block(self._shared_output)
            return self.mean_layer(r3), self.log_std_parameter, {}
        elif role == "value":
            # shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            state_data1 = (
                torch.cat(
                    [
                        (space["joint_pos"] - self.joint_pos_mean) / self.joint_pos_std,
                        (space["joint_vel"] - self.joint_vel_mean) / self.joint_vel_std,
                        # space["ee_position"],
                        # space["target_object_position"],
                        (space["actions"] - self.actions_mean) / self.actions_std,
                        # space["contact_force_left_finger"],
                        # space["contact_force_left_finger"],
                    ],
                    dim=-1,
                )
                .to(torch.float32)
                .clamp(max=10, min=-10)
            )
            state_data2 = (
                torch.cat(
                    [
                        # space["joint_pos"],
                        # space["joint_vel"],
                        (space["object_position"] - self.object_position_mean) / self.object_position_std,  # 3
                        (space["ee_position"] - self.ee_position_mean) / self.ee_position_std,  # 3
                        (space["target_object_position"] - self.target_object_position_mean)
                        / self.target_object_position_std,  # 7
                        # space["actions"],
                        space["contact_force_left_finger"],  # 1
                        space["contact_force_right_finger"],  # 1
                    ],
                    dim=-1,
                )
                .to(torch.float32)
                .clamp(max=10, min=-10)
            )

            feature1 = self.state_net1(state_data1)
            feature3 = self.state_net2(state_data2)
            r1 = self.block1(torch.cat([feature1, feature3], dim=-1))
            r2 = self.block2(r1)
            shared_output = r2
            return self.value_layer(shared_output), {}
