import ctypes
import json
import multiprocessing

import numpy as np
import pytest
from scipy import sparse

import lightgbm as lgb
from lightgbm.basic import _LIB, _c_str, _safe_call

from .utils import find_random_open_port


def _construct_distributed_dataset(data, params, machines, port, output):
    _safe_call(_LIB.LGBM_NetworkInit(_c_str(machines), ctypes.c_int(port), ctypes.c_int(1), ctypes.c_int(2)))
    try:
        lgb.Dataset(data, params=params)._dump_text(output)
    finally:
        _safe_call(_LIB.LGBM_NetworkFree())


@pytest.mark.parametrize("input_type", ["dense", "csr", "file"])
@pytest.mark.parametrize("per_feature_max_bin", [False, True])
@pytest.mark.parametrize("forced_bins", [False, True])
def test_distributed_bin_feature_indices(tmp_path, input_type, per_feature_max_bin, forced_bins):
    # Both workers receive identical data so that sampling cannot explain any
    # difference from local binning. Worker 1 owns features 2 and 3, whose
    # ranges and configuration differ from features 0 and 1.
    values = np.arange(1, 129, dtype=np.float64)
    data = np.column_stack([values + 1000 * feature for feature in range(4)])
    params = {
        "num_threads": 1,
        "max_bin": 7,
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "pre_partition": True,
        "verbosity": -1,
    }
    if per_feature_max_bin:
        params["max_bin_by_feature"] = [4, 5, 6, 7]
    if forced_bins:
        bounds = [
            {"feature": feature, "bin_upper_bound": [1000 * feature + 32.5, 1000 * feature + 96.5]}
            for feature in range(4)
        ]
        bounds_file = tmp_path / "forced_bins.json"
        bounds_file.write_text(json.dumps(bounds))
        params["forcedbins_filename"] = str(bounds_file)

    if input_type == "csr":
        data = sparse.csr_matrix(data)
    elif input_type == "file":
        data_file = tmp_path / "data.csv"
        np.savetxt(data_file, np.column_stack([np.zeros(len(values)), data]), delimiter=",")
        data = str(data_file)

    expected_file = tmp_path / "local.txt"
    lgb.Dataset(data, params=params)._dump_text(expected_file)
    expected = expected_file.read_text()

    ports = {find_random_open_port()}
    while len(ports) < 2:
        ports.add(find_random_open_port())
    machines = ",".join(f"127.0.0.1:{port}" for port in ports)
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for rank, port in enumerate(ports):
            # Windows adapter enumeration does not include the loopback address.
            # Set the rank explicitly for these local workers.
            worker = context.Process(
                target=_construct_distributed_dataset,
                args=(data, params, f"rank={rank},{machines}", port, tmp_path / f"rank{rank}.txt"),
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive(), "Distributed Dataset construction timed out"
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)

    for rank in range(2):
        assert (tmp_path / f"rank{rank}.txt").read_text() == expected
