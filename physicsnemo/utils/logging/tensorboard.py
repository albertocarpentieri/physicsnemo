# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Optional

from .console import PythonLogger
from .launch import LaunchLogger
from physicsnemo.distributed import DistributedManager

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


def initialize_tensorboard(log_dir: Optional[str] = None, use_tensorboard: bool = True) -> None:
    """Initializes TensorBoard writer and hooks it into LaunchLogger.

    Parameters
    ----------
    log_dir : str, optional
        Log directory for TensorBoard runs, defaults to "./runs" if not provided
    use_tensorboard : bool, optional
        Toggle whether to enable tensorboard, defaults to True
    """
    if not use_tensorboard:
        return

    dist = DistributedManager()
    if dist.rank != 0:
        # Only create a writer on root; toggle backend for consistency
        LaunchLogger.toggle_tensorboard(True)
        return

    if SummaryWriter is None:
        PythonLogger().warning("TensorBoard not available (install tensorboard). Turning off TB logging")
        return

    LaunchLogger.toggle_tensorboard(True)
    try:
        LaunchLogger.tb_writer = SummaryWriter(log_dir=log_dir or "./runs")
        PythonLogger().info(f"TensorBoard writer initialized at {log_dir or './runs'}")
    except Exception as e:
        PythonLogger().warning(f"Failed to initialize TensorBoard writer: {e}")
        LaunchLogger.toggle_tensorboard(False)
