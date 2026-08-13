# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from nemo_rl.models.automodel.draft.integration import (
    validate_dspark_checkpoint_meta,
)

_TRAINING_META = {
    "model_name": "deepseek-ai/dspark_qwen3_8b_block7",
    "block_size": 7,
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "optimizer_layout": [
        {"name": "policy", "num_params": 4},
        {"name": "draft", "num_params": 2},
    ],
}


def test_weights_only_load_skips_optimizer_layout():
    """init_optimizer=False loads (eval/logprob workers) carry no expected
    layout and must accept a checkpoint saved from training."""
    expected = dict(_TRAINING_META, optimizer_layout=None)
    validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_optimizer_resume_rejects_layout_mismatch():
    expected = dict(
        _TRAINING_META,
        optimizer_layout=[
            {"name": "draft", "num_params": 2},
            {"name": "policy", "num_params": 4},
        ],
    )
    with pytest.raises(ValueError, match="optimizer_layout"):
        validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_optimizer_resume_rejects_stale_unnamed_layout():
    saved = dict(
        _TRAINING_META, optimizer_layout=[{"num_params": 4}, {"num_params": 2}]
    )
    with pytest.raises(ValueError, match="optimizer_layout"):
        validate_dspark_checkpoint_meta(saved, _TRAINING_META)


def test_architecture_mismatch_always_rejected():
    expected = dict(_TRAINING_META, block_size=4, optimizer_layout=None)
    with pytest.raises(ValueError, match="block_size"):
        validate_dspark_checkpoint_meta(_TRAINING_META, expected)
