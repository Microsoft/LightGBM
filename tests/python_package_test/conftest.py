import numpy as np
import pytest


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="function")
def rng_fixed_seed():
    return np.random.default_rng(seed=42)
