import os

import pytest
import joblib
import time


@pytest.fixture(autouse=True)
def footer_function_scope(capsys):
    """Report test durations after each function."""
    start = time.time()
    yield
    stop = time.time()
    delta = stop - start
    with capsys.disabled():
        print("\ntest duration : {:0.3} seconds".format(delta))


@pytest.fixture()
def sample_output(capsys):
    """Load prediction output dictionary"""
    with open("./tests/assets/predict_out_sample.joblib", "rb") as fp:
        out = joblib.load(fp)
    return out


@pytest.fixture()
def change_working_dir():
    """Change working directory for loading resources"""
    import os

    os.chdir(
        "../model_ffm/"
    )  # Needed for proper resolving of data_dir variable in load_resources function
