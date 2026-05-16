import pytest
import numpy as np
from examples.ma2.model import prior, simulator, SUMMARY_FUNCTIONS
from abclib.statistics.handcrafted import HandCraftedSummary
from abclib.utils import run_pilot
from abclib.distance import euclidean

@pytest.fixture
def ma2_components():
    np.random.seed(42)
    stat = HandCraftedSummary(SUMMARY_FUNCTIONS)
    thetas, sims = run_pilot(prior, simulator, n_pilot=500)
    stat.fit(thetas, sims)
    return prior, simulator, stat, euclidean