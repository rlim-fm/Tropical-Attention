import importlib

import torch

import tropical_gemm_backend as tb


def test_missing_tropical_gemm_is_imported_once(monkeypatch):
    attempts = []
    real_import = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name.startswith('tropical_gemm'):
            attempts.append(name)
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(tb.importlib, 'import_module', import_module)
    monkeypatch.delenv(tb._BACKEND_ENV, raising=False)
    tb.reset_backend_cache()
    try:
        a, b = torch.randn(3, 4, 5), torch.randn(5, 6)
        for _ in range(5):
            assert torch.equal(tb.maxplus_matmul(a, b), tb.torch_maxplus_matmul(a, b))
        assert sorted(attempts) == ['tropical_gemm', 'tropical_gemm.pytorch']
        assert 'unavailable' in tb.maxplus_backend_status()
    finally:
        tb.reset_backend_cache()
