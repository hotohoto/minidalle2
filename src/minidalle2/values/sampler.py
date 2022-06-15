class SkippedSampler:
    def __init__(self, sampler, n_skip=0):
        self.sampler = sampler
        self.iter = iter(sampler)
        self.n_skip = n_skip

    def __iter__(self):
        for i in range(self.sampler.__len__()):
            if i < self.n_skip:
                next(self.iter)
            else:
                yield next(self.iter)

    def __len__(self):
        return max(self.sampler.__len__() - self.n_skip, 0)
