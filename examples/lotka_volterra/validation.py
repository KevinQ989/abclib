from .model import LotkaVolterra
from ..validation import run_validation
from ..config import (
    Config,
    RejectionConfig,
    SMCConfig,
    MCMCConfig,
    SyntheticLikelihoodConfig,
    SBCConfig,
    PPCConfig,
    STRConfig
)
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """Run the full abclib validation pipeline on the Lotka-Volterra model."""
    np.random.seed(0)
    model = LotkaVolterra(T=100)
    true_theta = np.array([0.5, 0.1, 0.1, 0.5])
    observed_data = model.simulator(true_theta)
    
    config = Config(
        methods = ["all"],
        n_pilot = 2_000,
        output_dir = OUTPUT_DIR,
        rejection = RejectionConfig(
            n_simulations=10_000, q=0.05
        ),
        smc = SMCConfig(
            M=10_000, T=5, q=0.05
        ),
        mcmc = MCMCConfig(
            n_samples=10_000, epsilon=1.0, proposal_std=0.05
        ),
        synthetic_likelihood=SyntheticLikelihoodConfig(
            n_simulations=2_000, M=200, proposal_std=0.001
        ),
        sbc=SBCConfig(
            n_trials=200, L=100
        ),
        ppc=PPCConfig(
            n_samples=1000
        ),
        str_=STRConfig(
            n_grid_points=3,
            credible_mass=0.90
        )
    )

    results = run_validation(
        model = model,
        config = config,
        true_theta = true_theta,
        observed_data = observed_data
    )


if __name__ == "__main__":
    main()