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
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.state_net1 = nn.Sequential(
            nn.Linear(26, 128),  # ik rel input 37，原38
            nn.BatchNorm1d(128),
            nn.ELU(),
            # nn.Linear(64, 16),
            # nn.ELU(),
        )
        self.state_net2 = nn.Sequential(
            nn.Linear(15, 128),  # ik rel input 37，原38
            nn.BatchNorm1d(128),
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
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Sequential(
            nn.Linear(256, 1),
        )

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):

        space = self.tensor_to_space(inputs["states"], self.observation_space)
        if role == "policy":

            state_data1 = torch.cat(
                [
                    space["joint_pos"],
                    space["joint_vel"],
                    # space["ee_position"],
                    # space["target_object_position"],
                    space["actions"],
                    # space["contact_force_left_finger"],
                    # space["contact_force_left_finger"],
                ],
                dim=-1,
            ).to(torch.float32)
            state_data2 = torch.cat(
                [
                    # space["joint_pos"],
                    # space["joint_vel"],
                    space["object_position"],  # 3
                    space["ee_position"],  # 3
                    space["target_object_position"],  # 7
                    # space["actions"],
                    space["contact_force_left_finger"],  # 1
                    space["contact_force_left_finger"],  # 1
                ],
                dim=-1,
            ).to(torch.float32)

            feature1 = self.state_net1(state_data1)
            feature3 = self.state_net2(state_data2)

            r1 = self.block1(torch.cat([feature1, feature3], dim=-1))
            r2 = self.block2(r1)
            self._shared_output = r2
            r3 = self.action_block(self._shared_output)
            return self.mean_layer(r3), self.log_std_parameter, {}
        elif role == "value":
            # shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            if self._shared_output is None:
                state_data1 = torch.cat(
                    [
                        space["joint_pos"],
                        space["joint_vel"],
                        # space["ee_position"],
                        # space["target_object_position"],
                        space["actions"],
                        # space["contact_force_left_finger"],
                        # space["contact_force_left_finger"],
                    ],
                    dim=-1,
                ).to(torch.float32)
                state_data2 = torch.cat(
                    [
                        # space["joint_pos"],
                        # space["joint_vel"],
                        space["object_position"],  # 3
                        space["ee_position"],  # 3
                        space["target_object_position"],  # 7
                        # space["actions"],
                        space["contact_force_left_finger"],  # 1
                        space["contact_force_left_finger"],  # 1
                    ],
                    dim=-1,
                ).to(torch.float32)

                feature1 = self.state_net1(state_data1)
                feature3 = self.state_net2(state_data2)
                r1 = self.block1(torch.cat([feature1, feature3], dim=-1))
                r2 = self.block2(r1)
                shared_output = r2
            else:
                shared_output = self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
