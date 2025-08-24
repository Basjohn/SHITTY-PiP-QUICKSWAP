import time
from utils.resource_manager import get_resource_manager, ResourceType


class _Dummy:
    """Simple weakref-able dummy resource."""
    pass


def test_snapshot_reflects_register_unregister():
    rm = get_resource_manager()

    obj = _Dummy()
    rid = rm.register(
        resource=obj,
        resource_type=ResourceType.CUSTOM,
        description="snapshot_test",
    )
    assert isinstance(rid, str) and rid

    # Allow worker to process and publish snapshot
    time.sleep(0.05)
    ids1 = {ri.resource_id for ri in rm.list_resources_snapshot()}
    assert rid in ids1

    # Unregister and validate snapshot reflects removal
    assert rm.unregister(rid, force=True)
    time.sleep(0.05)
    ids2 = {ri.resource_id for ri in rm.list_resources_snapshot()}
    assert rid not in ids2
