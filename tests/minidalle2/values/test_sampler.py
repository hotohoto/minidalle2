from torch.utils.data import BatchSampler, SequentialSampler

from minidalle2.values.sampler import SkippedSampler


class TestCustomBatchSampler:
    @staticmethod
    def test_skipped_sampler():
        s = SkippedSampler(
            BatchSampler(SequentialSampler(range(0, 10)), batch_size=3, drop_last=False), n_skip=1
        )
        assert len(s) == 3
        x = list(s)
        assert x == [[3, 4, 5], [6, 7, 8], [9]]
