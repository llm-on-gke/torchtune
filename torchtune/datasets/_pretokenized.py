# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._utils import truncate


class PretokenizedDataset(Dataset):
    """
    A dataset that assumes the data is already tokenized. It expects the dataset
    to have "input_ids" and "labels" columns.

    Args:
        source (str): path to dataset repository on Hugging Face.
        max_seq_len (Optional[int]): maximum sequence length. Default is None.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self._data[index]
        tokens = sample["input_ids"]
        labels = sample["labels"]
        if self.max_seq_len:
            tokens = truncate(tokens, self.max_seq_len)
            labels = truncate(labels, self.max_seq_len)
        return {"tokens": tokens, "labels": labels}


def pretokenized_dataset(
    tokenizer: Optional[Any] = None,
    *,
    source: str,
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: dict[str, Any],
) -> PretokenizedDataset:
    """
    Build a pre-tokenized dataset. This assumes that the data has already been
    tokenized and is structured in a way that can be directly used for training.
    The dataset should have "input_ids" and "labels" columns.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        max_seq_len (Optional[int]): maximum sequence length. Default is None.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        PretokenizedDataset: the configured :class:`~torchtune.datasets.PretokenizedDataset`

    Example:
        >>> pretokenized_ds = pretokenized_dataset(source="some/pretokenized-dataset")
        >>> for batch in Dataloader(pretokenized_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    return PretokenizedDataset(
        source=source, max_seq_len=max_seq_len, **load_dataset_kwargs
    )