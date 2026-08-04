from __future__ import annotations

import pytest

from module.auto_rig.pose.factory import PoseProviderPool


def test_pose_provider_pool_loads_each_backend_once_and_releases_it() -> None:
    loads: list[str] = []

    class Provider:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    provider = Provider()

    def loader(key: str) -> object:
        loads.append(key)
        return provider

    pool = PoseProviderPool(provider_loader=loader)

    assert pool.resolve("sdpose") is provider
    assert pool.resolve("sdpose") is provider
    assert loads == ["sdpose"]

    pool.close()

    assert provider.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        pool.resolve("sdpose")


def test_pose_provider_pool_keeps_backends_separate() -> None:
    loads: list[str] = []

    def loader(key: str) -> object:
        loads.append(key)
        return object()

    pool = PoseProviderPool(provider_loader=loader)

    assert pool.resolve("sdpose") is not pool.resolve("detrpose")
    assert loads == ["sdpose", "detrpose"]
