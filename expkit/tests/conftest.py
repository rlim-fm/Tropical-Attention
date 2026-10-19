import os
import sys

import pytest

# Importable however pytest is invoked: the directory holding expkit/ on the path.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def campaigns_tmp(tmp_path, monkeypatch):
    """Every test writes its campaigns into its own tmp directory."""
    monkeypatch.setenv('EXPKIT_CAMPAIGNS', str(tmp_path / 'campaigns'))
    return tmp_path / 'campaigns'

