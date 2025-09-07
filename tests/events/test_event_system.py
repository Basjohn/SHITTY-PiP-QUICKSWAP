from core.events.event_system import EventSystem


def test_basic_subscribe_publish_order_by_priority():
    es = EventSystem()
    order = []

    def cb_a(evt):
        order.append("A")

    def cb_b(evt):
        order.append("B")

    # Higher numeric runs earlier; zero runs last
    es.subscribe("t.x", cb_b, priority=0)
    es.subscribe("t.x", cb_a, priority=5)

    es.publish("t.x")
    assert order == ["A", "B"]


def test_priority_ties_are_subscribe_order_stable():
    es = EventSystem()
    order = []

    def cb1(evt):
        order.append(1)

    def cb2(evt):
        order.append(2)

    # Same priority -> preserve subscribe order (Python sort is stable)
    es.subscribe("tie", cb1, priority=3)
    es.subscribe("tie", cb2, priority=3)
    es.publish("tie")
    assert order == [1, 2]


def test_ui_thread_dispatch_flag_routes_via_thread_manager(monkeypatch):
    es = EventSystem()
    calls = []

    from core.threading import ThreadManager

    def fake_run_on_ui_thread(fn, evt):
        calls.append(("ui", getattr(fn, "__name__", str(fn))))
        # Execute immediately to keep semantics
        fn(evt)

    monkeypatch.setattr(ThreadManager, "run_on_ui_thread", fake_run_on_ui_thread)

    def handler(evt):
        calls.append(("handler", evt.type))

    es.subscribe("ui.test", handler, dispatch_on_ui=True)
    es.publish("ui.test")

    # First, the UI dispatcher is used, then the handler runs
    assert calls[0][0] == "ui"
    assert calls[-1] == ("handler", "ui.test")


def test_wildcard_matching_and_specificity():
    es = EventSystem()
    record = []

    def wild(evt):
        record.append(("wild", evt.type))

    def specific(evt):
        record.append(("specific", evt.type))

    # Specific with higher prio should run before wildcard
    es.subscribe("window.*", wild, priority=0)
    es.subscribe("window.created", specific, priority=2)

    es.publish("window.created")

    assert record[0][0] == "specific"
    assert record[1][0] == "wild"


def test_filter_fn_blocks_handler_invocation():
    es = EventSystem()
    called = {"count": 0}

    def handler(evt):
        called["count"] += 1

    es.subscribe("f.x", handler, filter_fn=lambda e: False)
    es.publish("f.x")
    assert called["count"] == 0


def test_handler_exception_isolated_and_logged(caplog):
    es = EventSystem()
    seen = []

    def bad(evt):
        raise RuntimeError("boom")

    def good(evt):
        seen.append("ok")

    es.subscribe("err.x", bad, priority=5)
    es.subscribe("err.x", good, priority=0)

    with caplog.at_level("ERROR"):
        es.publish("err.x")

    # Ensure the good handler still ran and an error was logged
    assert "ok" in seen
    assert any("Error in event handler" in rec.message for rec in caplog.records)


def test_unsubscribe_during_publish_does_not_break_iteration():
    es = EventSystem()
    calls = []
    sub_ids = {}

    def remover(evt):
        calls.append("remover")
        # Unsubscribe the other handler while iterating
        es.unsubscribe(sub_ids["other"])  # should not break iteration

    def other(evt):
        calls.append("other")

    sub_ids["remover"] = es.subscribe("race", remover, priority=5)
    sub_ids["other"] = es.subscribe("race", other, priority=4)

    es.publish("race")

    # Current semantics: unsubscribed handler is not invoked in the same publish pass
    assert calls == ["remover"]

    # Subsequent publish should only call the remaining subscribed handler
    calls.clear()
    es.publish("race")
    assert calls == ["remover"]
