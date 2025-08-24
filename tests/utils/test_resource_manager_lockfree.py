import time
from utils.resource_manager import get_resource_manager, ResourceType

class _DummyResource:
    """Simple user-defined class instances are weakref-able."""
    pass

def test_resources_list_updates_on_register_unregister():
    rm = get_resource_manager()

    # Ensure starting list builds without error
    resources0 = rm.list_resources_snapshot()
    assert isinstance(resources0, list)

    # Register a dummy custom resource (must be weakref-able)
    res = _DummyResource()
    rid = rm.register(
        resource=res,
        resource_type=ResourceType.CUSTOM,
        description="test_custom_lockfree",
        tags={"test"},
    )
    assert isinstance(rid, str) and len(rid) > 0

    # Allow any async side-effects to complete (if applicable)
    time.sleep(0.05)
    ids1 = {ri.resource_id for ri in rm.list_resources_snapshot()}
    assert rid in ids1

    # Unregister and confirm list reflects removal
    assert rm.unregister(rid, force=True)
    time.sleep(0.05)
    ids2 = {ri.resource_id for ri in rm.list_resources_snapshot()}
    assert rid not in ids2
