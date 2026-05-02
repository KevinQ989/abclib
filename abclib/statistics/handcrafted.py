from .base import BaseSummaryStatistic

class HandCraftedSummary(BaseSummaryStatistic):
    def fit(self, thetas, simulations):
        raise NotImplementedError("HandCraftedSummary.fit not yet implemented.")
    

    def transform(self, simulation):
        raise NotImplementedError("HandCraftedSummary.transform not yet implemented.")