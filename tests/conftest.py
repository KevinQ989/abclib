import pytest
import numpy as np
from examples.ma2.model import MA2
from abclib.statistics.handcrafted import HandCraftedSummary
from abclib.utils import run_pilot
from abclib.distance import euclidean

@pytest.fixture
def ma2_components():
    np.random.seed(42)
    model = MA2(T=100)
    stat = HandCraftedSummary(model.SUMMARY_FUNCTIONS)
    thetas, sims = run_pilot(model.prior, model.simulator, n_pilot=500)
    stat.fit(thetas, sims)
    return model.prior, model.simulator, stat, euclidean