import bisect
import os

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from Lara.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

to_pil = T.ToPILImage()


def _boundary_state_from_sequence(state_sequence, boundary_index: int):
    valid = boundary_index < state_sequence.shape[0]
    safe_index = min(boundary_index, state_sequence.shape[0] - 1)
    return state_sequence[safe_index : safe_index + 1], bool(valid)


def collate_fn(
    batch,
    img_keys=None,
    state_key="observation.state",
    action_key="action",
    task_key="task",
    resize_size=224,
    execution_horizon=None,
    prediction_horizon=None,
):
    examples = []
    for _, b in enumerate(batch):
        example = {"image": []}
        img_keys = img_keys or []
        example["action"] = b[action_key].cpu().numpy()
        example["future_actions"] = example["action"]
        action_pad_key = f"{action_key}_is_pad"
        if action_pad_key in b:
            example["future_action_mask"] = (~b[action_pad_key].bool()).cpu().numpy()
        else:
            example["future_action_mask"] = torch.ones(b[action_key].shape[0], dtype=torch.bool).cpu().numpy()
        if task_key in b:
            example["lang"] = b[task_key]
        elif "task" in b:
            example["lang"] = b["task"]
        elif "task_index" in b:
            example["lang"] = f"task_{int(torch.as_tensor(b['task_index']).flatten()[0])}"
        else:
            example["lang"] = "perform the task"

        for k in img_keys:
            img_primary = to_pil(b[k][0]).resize((resize_size, resize_size))
            example["image"].append(img_primary)

        for k in b.keys():
            if k == state_key:
                state_sequence = b[k].cpu().numpy()
                example["state"] = state_sequence[0:1]
                example["current_state"] = example["state"]
                prediction_index = prediction_horizon if prediction_horizon is not None else state_sequence.shape[0] - 1
                execution_index = execution_horizon if execution_horizon is not None else prediction_index
                execution_state, execution_valid = _boundary_state_from_sequence(state_sequence, execution_index)
                prediction_state, prediction_valid = _boundary_state_from_sequence(state_sequence, prediction_index)
                example["execution_state_target"] = execution_state
                example["execution_state_target_mask"] = execution_valid
                example["prediction_state_target"] = prediction_state
                example["prediction_state_target_mask"] = prediction_valid
        examples.append(example)
    return examples

class MixtureDataset(Dataset):
    def __init__(self, datasets):
        """
        datasets: List[Dataset]
        """
        self.datasets = datasets
        # prefix sum of lengths，用于快速定位 index 属于哪个 dataset
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self):
        sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # 找到 idx 属于哪个 dataset
        ds_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if ds_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx][sample_idx]


def get_lerobot_v3_datasets(
    data_cfg: dict,
    action_horizon: int | None = None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    action_horizon = action_horizon if action_horizon is not None else data_cfg.get("action_horizon", 60)
    state_key = data_cfg.get("state_key", "observation.state")
    action_key = data_cfg.get("action_key", "action")
    img_keys_override = data_cfg.get("img_keys", None)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        repo_id = os.path.join(data_root_dir, d_name)
        ds_meta = LeRobotDatasetMetadata(repo_id)

        if img_keys_override:
            image_keys = list(img_keys_override)
        else:
            image_keys = [
                k
                for k, feature in ds_meta.features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "image"
            ]
        missing_img_keys = [k for k in image_keys if k not in ds_meta.features]
        if missing_img_keys:
            raise KeyError(f"Configured image keys are missing from {repo_id}: {missing_img_keys}")
        if action_key not in ds_meta.features:
            raise KeyError(f"Configured action key {action_key!r} is missing from {repo_id}")
        if state_key not in ds_meta.features:
            raise KeyError(f"Configured state key {state_key!r} is missing from {repo_id}")
        delta_timestamps = {
            # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
            action_key: [t / ds_meta.fps for t in range(action_horizon)],
            state_key: [t / ds_meta.fps for t in range(action_horizon + 1)],
        }
        for k in image_keys:
            delta_timestamps[k] = [t / ds_meta.fps for t in range(action_horizon+1)]
        dataset_mixture.append(
            LeRobotDataset(
                repo_id,
                delta_timestamps=delta_timestamps,
            )
    )
    #[print(ds.num_episodes, ds.num_frames, i) for i, ds in enumerate(dataset_mixture)]
    return MixtureDataset(dataset_mixture)
