from abclib.base import BaseSampler

class RejectionABC(BaseSampler):
    def __init__(self, prior, simulator, summary_statistic, distance):
        super().__init__(prior, simulator, summary_statistic, distance)
    
    def sample(self, s_obs, n_sample, **kwargs):
        pass