import quantum_hw.api as api


def test_api_exports_include_unified_symbols():
    expected = {
        "Backend",
        "QuantumHardwareClient",
        "ResolvedBackend",
        "HardwareTopology",
        "HardwareCalibration",
        "HardwareProfile",
        "BackendAdapter",
        "QuafuBackendAdapter",
        "TianYanBackendAdapter",
        "GuoDunBackendAdapter",
        "OpenQasmSubmitRequest",
        "ProviderTaskHandle",
        "TaskAdapter",
        "QuafuTaskAdapter",
        "TianYanTaskAdapter",
        "GuoDunTaskAdapter",
        "ProviderRuntime",
        "create_provider_runtime",
        "QuafuPlatform",
        "TianYanPlatform",
        "GuoDunPlatform",
        "QuantumLanguage",
    }
    assert expected.issubset(set(api.__all__))


def test_api_unified_symbols_are_accessible():
    assert api.Backend is not None
    assert api.ResolvedBackend is not None
    assert api.HardwareTopology is not None
    assert api.HardwareCalibration is not None
    assert api.HardwareProfile is not None
    assert api.BackendAdapter is not None
    assert api.QuafuBackendAdapter is not None
    assert api.TianYanBackendAdapter is not None
    assert api.GuoDunBackendAdapter is not None
    assert api.OpenQasmSubmitRequest is not None
    assert api.ProviderTaskHandle is not None
    assert api.TaskAdapter is not None
    assert api.QuafuTaskAdapter is not None
    assert api.TianYanTaskAdapter is not None
    assert api.GuoDunTaskAdapter is not None
    assert api.ProviderRuntime is not None
    assert api.create_provider_runtime is not None
    assert api.QuafuPlatform is not None
    assert api.TianYanPlatform is not None
    assert api.GuoDunPlatform is not None
    assert api.QuantumLanguage is not None