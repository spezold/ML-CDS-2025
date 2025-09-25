# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Iterator, Sequence
from operator import itemgetter
import numpy as np

from torch.utils.data import Dataset, Sampler, DistributedSampler


class DatasetFromSampler(Dataset):

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """ Convert any Pytorch Sampler to a DistributedSampler """

    def __init__(
        self,
        sampler,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class CustomWeightedRandomSampler(Sampler[int]):
    """ Generalized WeightedRandomSampler to allow for more than 2^24 samples """

    def __init__(self, weights: Sequence[float] | None, num_samples_available: int, num_samples_drawn: int):
        self.weights = weights
        self.num_samples_available = num_samples_available  # Total num. of samples available to the sampler
        self.num_samples_drawn = num_samples_drawn  # Num. of samples actually returned by the sampler == `len(sampler)`
        super().__init__()

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(self.num_samples_available),
            size=self.num_samples_drawn,
            p=None if self.weights is None else np.divide(self.weights, np.sum(self.weights), dtype=float),
            replace=self.num_samples_drawn > self.num_samples_available
        ).tolist()
        return iter(rand_tensor)

    def __len__(self):
        return self.num_samples_drawn


@dataclass(frozen=True, kw_only=True)
class DistributedWeightedSamplerArgs:
    number_type: str  # "relative" or "absolute"
    number_value: int | float
    num_replicas: int | None
    rank: int | None
    shuffle: bool


class DistributedWeightedSampler(DistributedSamplerWrapper):

    @staticmethod
    def _check_inputs_and_return_num_samples(weights, num_samples):
        if weights is None and num_samples is None:
            raise ValueError("No 'weights' are given, so 'num_samples' must be given (is None).")
        elif weights is not None and num_samples is not None and len(weights) != num_samples:
            raise ValueError(f"Mismatch: `len(weights)=={len(weights)}`, but `num_samples=={num_samples}` "
                             f"(hint: if 'weights' are given, 'num_samples' is optional).")
        return num_samples if num_samples is not None else len(weights)

    def __init__(
        self,
        args: DistributedWeightedSamplerArgs,
        weights: Sequence[float] | None,
        num_samples: int | None = None,
    ):
        num_samples_available = self._check_inputs_and_return_num_samples(weights, num_samples)
        if args.number_type == "absolute":
            num_samples_drawn = args.number_value
        elif args.number_type == "relative":
            num_samples_drawn = round(args.number_value * num_samples_available)
        else:
            raise ValueError(f"Unknown args.number_type: '{args.number_type}' (should be 'absolute' or 'relative')")
        weighted_sampler = CustomWeightedRandomSampler(
            weights=weights,
            num_samples_available=num_samples_available,
            num_samples_drawn=num_samples_drawn,
        )
        super().__init__(
            sampler=weighted_sampler,
            num_replicas=args.num_replicas,
            rank=args.rank,
            shuffle=args.shuffle,
        )
