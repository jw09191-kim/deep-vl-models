import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        skip = pytest.mark.skip(reason="HF_HUB_OFFLINE=1 — integration 테스트 건너뜀")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
