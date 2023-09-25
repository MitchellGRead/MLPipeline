import os

import numpy as np
import ray.data as ray_data
import torch
from ray.data import Dataset

from ml.api.data_handler_interface import (
    DataHandlerInterface,
    IndexableDataset,
)

from . import utils


class ComplexPhysicsDataHandler(DataHandlerInterface, IndexableDataset):
    def __init__(
        self,
        data_loc: str,
        num_samples: int = None,
        window_length: int = 7,
    ) -> None:
        if not os.path.isdir(data_loc):
            raise ValueError(
                f"ComplexPhysicsDataHandler expects a directory file path - {data_loc}"
            )

        self.metadata = utils.load_dict(os.path.join(data_loc, "metadata.json"))
        self.offset = utils.load_dict(os.path.join(data_loc, "test_offset.json"))
        self.offset = {int(k): v for k, v in self.offset.items()}  # convert dict key type to int
        self.window_length = window_length

        self.particle_type = np.memmap(
            os.path.join(data_loc, "test_particle_type.dat"), dtype=np.int64, mode="r"
        )

        self.position = np.memmap(
            os.path.join(data_loc, "test_position.dat"), dtype=np.float32, mode="r"
        )

        for traj in self.offset.values():
            self.dim = traj["position"]["shape"][2]

        # cut particle trajectories according to time slices
        self.windows = []
        for traj in self.offset.values():
            size = traj["position"]["shape"][1]
            length = traj["position"]["shape"][0] - window_length + 1
            for i in range(length):
                desc = {
                    "size": size,
                    "type": traj["particle_type"]["offset"],
                    "pos": traj["position"]["offset"] + i * size * self.dim,
                }
                self.windows.append(desc)

    def len(self) -> int:
        return len(self.windows)

    def get_data(self) -> Dataset:
        return ray_data.from_numpy(self.position)

    def get(self, idx: int) -> dict[str, any]:
        # load corresponding data for this time slice
        window = self.windows[idx]
        size = window["size"]

        particle_type = self.particle_type[
            window["type"] : window["type"] + size  # noqa: E203
        ].copy()
        particle_type = torch.from_numpy(particle_type)

        position_seq = self.position[
            window["pos"] : window["pos"] + self.window_length * size * self.dim  # noqa: E203
        ].copy()
        position_seq.resize(self.window_length, size, self.dim)  # list[list[list[float]]]
        position_seq = position_seq.transpose(1, 0, 2)

        target_position = position_seq[:, -1]
        position_seq = position_seq[:, :-1]

        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)

        """
        "particle_type": tensor [offset shape],
        "position_seq": tensor [offset shape, window_size - 1, dim],
        "target_position": tensor [offset shape, dim]
        """
        return {
            "particle_type": particle_type,
            "position_seq": position_seq,
            "target_position": target_position,
        }

    def split_data(
        self, test_size: float, shuffle: bool = True, seed: int = 1234
    ) -> tuple[Dataset, Dataset]:
        # TODO implement data splitting
        pass

    def add_to_config(self, train_loop_config: dict) -> dict:
        return super().add_to_config(train_loop_config)
