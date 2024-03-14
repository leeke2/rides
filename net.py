import gym
import numpy as np
import torch
from torch import Tensor, nn


class CNN(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_space, out_dim):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()

        if isinstance(obs_space, gym.spaces.Dict):
            obs_space = obs_space["obs"].shape

        self.conv = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        conv_out_size = self._get_conv_out(obs_space)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        idx_a, idx_b = np.triu_indices(obs_space[-1], k=1)
        self.idx_a = list(idx_a)
        self.idx_b = list(idx_b)

    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x) -> Tensor:
        """Forward pass through network.
        Args:
            x: input to network
        Returns:
            output of network
        """

        if isinstance(x, dict):
            batch_size = x["obs"].shape[0]

            if x["obs"].ndim == 4:
                conv_out = self.conv(x["obs"].float()).view(batch_size, -1)
            else:
                in_ = x["obs"].float()
                in1 = in_.new_zeros((*in_.shape[:-1], 45, 45))
                in1[:, :, self.idx_a, self.idx_b] = in_

                conv_out = self.conv(in1).view(batch_size, -1)
            # conv_out = self.conv(x["obs"]).view(batch_size, -1)
            q_values = self.head(conv_out)

        else:
            batch_size = x.shape[0]
            conv_out = self.conv(x).view(batch_size, -1)
            q_values = self.head(conv_out)

        if isinstance(x, dict) and "mask" in x:
            mask = torch.zeros_like(x["mask"], dtype=q_values.dtype)
            mask.masked_fill_(~x["mask"], float("-inf"))

            q_values += mask

        return q_values


def load_ckpt(ckpt_fn, net):
    ckpt = torch.load(
        ckpt_fn,
        map_location=torch.device("cpu"),
    )

    state_dict = {
        key.replace("target_net.", ""): val
        for key, val in ckpt["state_dict"].items()
        if key[:10] == "target_net"
    }

    net.load_state_dict(state_dict)


def to_tensor(x):
    return {key: torch.from_numpy(value).unsqueeze(0) for key, value in x.items()}
