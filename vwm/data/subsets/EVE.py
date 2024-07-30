import os

import torch

from .common import BaseDataset
import pickle
from pathlib import Path

import h5py
import torch
from loguru import logger
from ml_collections import config_dict
from PIL import Image
from streaming import StreamingDataset, base
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor
import numpy as np 
from gb_inference.data_row import Eve2DataRow
from gd.mds_key_reader import SliceStream



def date_from_log(log: str):
    """
    ailog_E2-046_2024_03_11-23_29_57 --> 2024_03_11
    """
    if not log.startswith("ailog_"):
        raise RuntimeError(f"Log name does not start with 'ailog_': {log}")
    return log[13:23]


def get_mds_root(is_ci: bool = False) -> Path:
    if is_ci:
        return Path("/tmp/featurized_data_mds")
    local_nas = os.environ.get("LOCAL_NAS", "/mnt/sunny_nas")
    return Path(local_nas) / "featurized_data_mds"


def paused(driving_command, joint_velocity_dict, total_joint_velocity_threshold=.15):
    # Determine whether to filter out frame as pause between enabling continuous transmission and grabbing clutch
    if driving_command is None and joint_velocity_dict is not None:
        total_joint_vels = np.sum(np.abs(list(joint_velocity_dict.values())))
        if total_joint_vels < total_joint_velocity_threshold:
            return True
    return False



class EveDataset(BaseDataset):
    def __init__(
        self,
        h5_file,
        video_stride = 6,
        target_height=256, 
        target_width=256, 
        num_frames=15,
        remove_pauses=True,
    ):
        """
        prediction_stride: number of frames between predictions in the sequence, minimal possible stride
        video_stride: number of frames between video frames in the sequence, has to be a multiple of prediction_stride
        """
        self.video_seq_len = num_frames
        self.video_stride = video_stride
        self.remove_pauses = remove_pauses

        self.hf = h5py.File(str(h5_file), "r")
        self.log_pts_map = pickle.loads(self.hf["log_pts_map"][()])

        self.n_sequences, self.dataset_idx2seg_map = self._compute_len()
        logger.info(f"Dataset has {self.n_sequences} sequences")

        # Initialize MDS dataset
        # NOTE: we don't use MDS shuffling functionality, because we index it directly
        # NOTE: creating a subdataset is a hack, but it's needed because if you inherit
        # from it we become an Iterable dataset and at the end of the epoch we get an indexing error
        # TODO: switch to reading frames directly via `mds_shard_num` and `mds_shard_idx`
        self.mds_subdataset = StreamingDataset(streams=self._get_mds_steams(), allow_unsafe_types=True)

    

        super().__init__(self.hf, None, target_height, target_width, num_frames)
        print("YouTube loaded:", len(self))


    def get_image_path(self, data_row, expected_pts):
        f_span_idx, mds_idx = self.log_pts_map[(data_row.log, data_row.pts)][0]
        row = self.mds_subdataset[mds_idx]
        if expected_pts is not None and row["pts"] != expected_pts:
            raise ValueError(f"Expected pts to be {expected_pts}, got {row['pts']}")
        return row["image_left"]


    def build_data_dict(self, image_seq):
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug
        }
        return data_dict

    def _get_mds_steams(self):
        # We assume all data is local, because we use MDS via direct indexing instead of Iterable dataset
        mds_root = get_mds_root(is_ci=False)
        shard_masks = pickle.loads(self.hf["shard_masks"][()])

        streams = []
        n_missing_logs = 0
        log_names = pickle.loads(self.hf["metadata"][2])
        log_names = sorted(log_names)  # required for correct indexing between h5 and MDS
        for log_name in log_names:
            local_path = f"{mds_root}/{date_from_log(log_name)}/{log_name}/"

            if not os.path.exists(local_path):
                n_missing_logs += 1
                logger.warning(f"{local_path} does not exist, skipping")
                # if n_missing_logs > self.max_logs_missing:
                #     raise RuntimeError(f"Too many missing logs (>{self.max_logs_missing})")
                continue

            stream = SliceStream(local=local_path, shard_mask=shard_masks[log_name], repeat=1)
            streams.append(stream)

        if len(streams) == 0:
            raise RuntimeError("All logs are missing")
        return streams

    def _add_sequence(self, item, start_frame_idx, n_frames_in_seq):
        for i in range(n_frames_in_seq ):
            eve_row = pickle.loads(item[start_frame_idx + i ])
            # Filter out the sequence if it has a frame with a pause
            if self.remove_pauses and paused(eve_row.driving_command, eve_row.joint_velocities):
                return False
        return True

    def _compute_len(self):
        """
        Guarentees that for every dataset idx there will be enough
        future frames to form a sequence of length `video_seq_len`
        with a given stride.

        Note that we do not check that we have enough frames for predictions
        of the last frame, because we want to learn to predict zero actions
        in the end of the segment execution (at least current code does this)
        """
        n_segments = sum(int(key.isdigit()) for key in self.hf.keys())
        n_sequences = 0
        n_skipped_sequences = 0
        total_possible_sequences = 0
        n_skipped_segments = 0

        n_frames_in_seq = self.video_stride * self.video_seq_len
        dataset_idx2seg_map = {}
        for i in range(n_segments):
            item = self.hf[str(i)]

            # there are this many sequences that have enough frames
            # to make a full sequence with a given stride
            _n_seqs = len(item) - n_frames_in_seq
            total_possible_sequences += _n_seqs  # in the future this might be different from n_sequences

            if _n_seqs == 0:
                # these sequences break frame_idx2seg_map and indexing within `segments`
                n_skipped_segments += 1
                _log_name = pickle.loads(item[0]).log
                print(f"Skipping a segment from {_log_name} as it only has {len(item)} frames")
                continue

            seq_idx = 0
            for start_frame_idx in range(_n_seqs):
                if self._add_sequence(item, start_frame_idx, n_frames_in_seq):
                    dataset_idx2seg_map[n_sequences + seq_idx] = (i, start_frame_idx)  # segment_idx, segment_start_idx
                    seq_idx += 1
                else:
                    n_skipped_sequences += 1
            n_sequences += seq_idx

        print(f"Skipped {n_skipped_sequences} sequences out of {total_possible_sequences} total")
        print(f"Skipped {n_skipped_segments} segments out of {n_segments} total")
        return n_sequences, dataset_idx2seg_map

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        segment_idx, frame_start_idx = self.dataset_idx2seg_map[idx]
        # now we need to get the index within the segment

        # --- Data extraction
        item = self.hf[str(segment_idx)]
        data_rows = []
        for idx in range(0, self.video_seq_len * self.video_stride, self.video_stride):
            _local_frame_idx = frame_start_idx + idx
            data_row: Eve2DataRow = pickle.loads(item[_local_frame_idx])
            data_rows.append(data_row)

        # --- Inputs Featurization
        image_seq = list()
        time_stamps = []
        f_captions = []
        # time_deltas = []
        # prev_time = data_rows[0].rtpc_monotonic_time
        for i in range(0, len(data_rows)):
            data_row = data_rows[i]
            img_path = self.get_image_path(data_row, expected_pts=data_row.pts)
            image = self.preprocess_image(img_path)
            image_seq.append(image)
            # time_stamps.append(data_row.train_metadata["frame_sec"])
            # f_captions.append(data_row.train_metadata["f_caption"])
        return self.build_data_dict(image_seq)