import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from module.see_through.model_manager import SeeThroughModelManager


def test_release_layerdiff_delete_policy_clears_reference():
    manager = SeeThroughModelManager(offload_policy="delete")
    manager._layerdiff = object()

    manager.release_layerdiff()

    assert manager._layerdiff is None


def test_release_all_clears_phase_references():
    manager = SeeThroughModelManager(offload_policy="delete")
    manager._layerdiff = object()
    manager._marigold = object()

    manager.release_all()

    assert manager._layerdiff is None
    assert manager._marigold is None
