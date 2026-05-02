from .base import BaseSummaryStatistic

class SemiAutomaticSummary(BaseSummaryStatistic):
    def fit(self, thetas, simulations):
        raise NotImplementedError("SemiAutomaticSummary.fit not yet implemented.")
    

    def transform(self, simulation):
        raise NotImplementedError("SemiAutomaticSummary.transform not yet implemented.")