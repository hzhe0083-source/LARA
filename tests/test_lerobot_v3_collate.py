import unittest
import tempfile
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Lara.dataloader import build_dataloader
from Lara.dataloader.lerobot_v3_datasets import TaskTextDataset, _load_task_map, collate_fn, get_lerobot_v3_datasets


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class LeRobotV3CollateTest(unittest.TestCase):
    def test_collate_exposes_explicit_future_actions_and_current_state(self):
        batch = [
            {
                "observation.image": torch.zeros(2, 3, 4, 4),
                "observation.state": torch.arange(6, dtype=torch.float32).view(2, 3),
                "action": torch.ones(3, 2),
                "action_is_pad": torch.tensor([False, False, True]),
                "task": "pick",
            }
        ]

        examples = collate_fn(
            batch,
            img_keys=["observation.image"],
            state_key="observation.state",
            action_key="action",
            task_key="task",
            resize_size=8,
            video_resolution_size=6,
            video_horizon=2,
            execution_horizon=1,
            prediction_horizon=2,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["lang"], "pick")
        self.assertEqual(len(examples[0]["image"]), 1)
        self.assertEqual(examples[0]["video"].shape, (1, 2, 6, 6, 3))
        self.assertEqual(examples[0]["video"].dtype, np.uint8)
        self.assertTrue(np.array_equal(examples[0]["future_actions"], examples[0]["action"]))
        self.assertTrue(np.array_equal(examples[0]["future_action_mask"], np.array([True, True, False])))
        self.assertTrue(np.array_equal(examples[0]["current_state"], examples[0]["state"]))
        self.assertEqual(examples[0]["current_state"].shape, (1, 3))
        self.assertTrue(np.array_equal(examples[0]["execution_state_target"], np.array([[3, 4, 5]], dtype=np.float32)))
        self.assertTrue(np.array_equal(examples[0]["prediction_state_target"], np.array([[3, 4, 5]], dtype=np.float32)))
        self.assertTrue(examples[0]["execution_state_target_mask"])
        self.assertFalse(examples[0]["prediction_state_target_mask"])

    def test_collate_pads_short_video_sequence_to_vj2_horizon(self):
        batch = [
            {
                "observation.image": torch.zeros(2, 3, 4, 4),
                "observation.wrist": torch.ones(2, 3, 4, 4),
                "observation.state": torch.arange(6, dtype=torch.float32).view(2, 3),
                "action": torch.ones(3, 2),
                "task": "pick",
            }
        ]

        examples = collate_fn(
            batch,
            img_keys=["observation.image", "observation.wrist"],
            state_key="observation.state",
            action_key="action",
            task_key="task",
            resize_size=8,
            video_resolution_size=5,
            video_horizon=4,
            execution_horizon=1,
            prediction_horizon=2,
        )

        self.assertEqual(examples[0]["video"].shape, (2, 4, 5, 5, 3))
        self.assertTrue(np.array_equal(examples[0]["video"][0, 2], examples[0]["video"][0, 3]))
        self.assertTrue(np.array_equal(examples[0]["video"][1, 2], examples[0]["video"][1, 3]))

    def test_task_text_wrapper_maps_task_index_before_collate(self):
        class DummyDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {
                    "observation.image": torch.zeros(2, 3, 4, 4),
                    "observation.state": torch.arange(6, dtype=torch.float32).view(2, 3),
                    "action": torch.ones(3, 2),
                    "task_index": torch.tensor([7]),
                }

        dataset = TaskTextDataset(DummyDataset(), {7: "open the drawer"})
        examples = collate_fn(
            [dataset[0]],
            img_keys=["observation.image"],
            state_key="observation.state",
            action_key="action",
            task_key="task",
            resize_size=8,
            execution_horizon=1,
            prediction_horizon=2,
        )

        self.assertEqual(examples[0]["lang"], "open the drawer")

    def test_load_task_map_accepts_lerobot_v3_index_level_task_text(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "meta"
            meta_dir.mkdir()
            pq.write_table(
                pa.table(
                    {
                        "task_index": [7],
                        "__index_level_0__": [
                            "LIVING_ROOM_SCENE2: put both the alphabet soup and the tomato sauce in the basket"
                        ],
                    }
                ),
                meta_dir / "tasks.parquet",
            )

            task_map = _load_task_map(tmpdir)

        self.assertEqual(
            task_map[7],
            "LIVING_ROOM_SCENE2: put both the alphabet soup and the tomato sauce in the basket",
        )

    def test_build_dataloader_passes_vj2_video_horizon_to_v3_dataset_and_collate(self):
        class TinyDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {
                    "observation.image": torch.zeros(3, 3, 4, 4),
                    "observation.state": torch.zeros(4, 2),
                    "action": torch.zeros(5, 1),
                    "task": "pick",
                }

        cfg = SimpleNamespace(
            framework=SimpleNamespace(
                action_model=AttrDict({
                    "action_horizon": 5,
                    "execution_horizon": 2,
                }),
                vj2_model=SimpleNamespace(num_frames=3),
            ),
            datasets=SimpleNamespace(
                vla_data=AttrDict({
                    "per_device_batch_size": 1,
                    "img_keys": ["observation.image"],
                    "state_key": "observation.state",
                    "action_key": "action",
                    "task_key": "task",
                    "resize_size": 8,
                    "video_resolution_size": 6,
                })
            ),
        )
        dataset_calls = []

        def fake_get_lerobot_v3_datasets(**kwargs):
            dataset_calls.append(kwargs)
            return TinyDataset()

        with patch("Lara.dataloader.lerobot_v3_datasets.get_lerobot_v3_datasets", fake_get_lerobot_v3_datasets):
            dataloader = build_dataloader(
                cfg,
                dataset_py="lerobot_v3_datasets",
                data_cfg=cfg.datasets.vla_data,
                mode="val",
                shuffle=False,
            )

        self.assertEqual(dataset_calls[0]["action_horizon"], 5)
        self.assertEqual(dataset_calls[0]["video_horizon"], 3)
        self.assertEqual(dataloader.collate_fn.keywords["video_horizon"], 3)
        self.assertEqual(dataloader.collate_fn.keywords["video_resolution_size"], 6)
        examples = dataloader.collate_fn([TinyDataset()[0]])
        self.assertEqual(examples[0]["video"].shape, (1, 3, 6, 6, 3))
        self.assertEqual(examples[0]["future_actions"].shape, (5, 1))

    def test_get_lerobot_v3_uses_video_horizon_for_image_delta_timestamps(self):
        dataset_calls = []

        class FakeMetadata:
            fps = 30
            features = {
                "observation.image": {"dtype": "image"},
                "observation.wrist": {"dtype": "image"},
                "observation.state": {"shape": [9]},
                "action": {"shape": [7]},
            }

            def __init__(self, repo_id):
                self.repo_id = repo_id

        class FakeLeRobotDataset:
            def __init__(self, repo_id, delta_timestamps):
                dataset_calls.append({"repo_id": repo_id, "delta_timestamps": delta_timestamps})

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                raise IndexError

        fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
        fake_module.LeRobotDataset = FakeLeRobotDataset
        fake_module.LeRobotDatasetMetadata = FakeMetadata
        data_cfg = AttrDict(
            {
                "data_root_dir": "/tmp/benchmarks/libero100",
                "data_mix": "libero100",
                "img_keys": ["observation.image", "observation.wrist"],
                "state_key": "observation.state",
                "action_key": "action",
            }
        )

        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": fake_module}):
            dataset = get_lerobot_v3_datasets(data_cfg, action_horizon=60, video_horizon=8)

        self.assertEqual(len(dataset), 1)
        delta = dataset_calls[0]["delta_timestamps"]
        self.assertEqual(len(delta["action"]), 60)
        self.assertEqual(len(delta["observation.state"]), 61)
        self.assertEqual(len(delta["observation.image"]), 8)
        self.assertEqual(len(delta["observation.wrist"]), 8)
        self.assertEqual(delta["observation.image"][-1], 7 / 30)


if __name__ == "__main__":
    unittest.main()
