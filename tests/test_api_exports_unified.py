import quantum_hw.api as api


def test_api_exports_include_unified_symbols():
    expected = {
        "Backend",
        "QuantumHardwareClient",
        "rank_chips",
        "ResolvedBackend",
        "BackendAdapter",
        "QuafuBackendAdapter",
        "CqlibBackendAdapter",
        "TaskRequest",
        "TaskAdapter",
        "QuafuTaskAdapter",
        "CqlibTaskAdapter",
    }
    assert expected.issubset(set(api.__all__))


def test_api_unified_symbols_are_accessible():
    assert api.ResolvedBackend is not None
    assert api.BackendAdapter is not None
    assert api.QuafuBackendAdapter is not None
    assert api.CqlibBackendAdapter is not None
    assert api.TaskRequest is not None
    assert api.TaskAdapter is not None
    assert api.QuafuTaskAdapter is not None
    assert api.CqlibTaskAdapter is not None
