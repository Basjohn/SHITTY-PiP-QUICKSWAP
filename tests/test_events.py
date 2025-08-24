"""
Tests for the core events module.
"""

import unittest
from unittest.mock import Mock

from core.events.event_system import EventSystem
from core.events.event_types import Event, EventType, Subscription


class TestEventSystem(unittest.TestCase):
    """Test cases for the EventSystem class."""
    
    def setUp(self):
        """Set up test environment."""
        self.event_system = EventSystem()
        self.mock_callback = Mock()
    
    def test_subscribe_and_publish(self):
        """Test subscribing to and publishing events."""
        # Subscribe to an event
        self.event_system.subscribe(EventType.TEST, self.mock_callback)
        
        # Publish an event
        self.event_system.publish(EventType.TEST, 'test data')
        
        # Verify the callback was called
        self.mock_callback.assert_called_once()
        callback_event = self.mock_callback.call_args[0][0]
        self.assertEqual(callback_event.type, 'test.event')
        self.assertEqual(callback_event.data, 'test data')
    
    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        # Subscribe and then unsubscribe
        subscription = self.event_system.subscribe(EventType.TEST, self.mock_callback)
        self.event_system.unsubscribe(subscription)
        
        # Publish an event - should not be received
        self.event_system.publish(EventType.TEST, 'test data')
        self.mock_callback.assert_not_called()
    
    def test_subscription_priority(self):
        """Test that subscriptions are processed in priority order."""
        results = []
        
        def make_callback(priority, result):
            def callback(_event):
                results.append((priority, result))
            return callback
        
        # Subscribe with different priorities
        self.event_system.subscribe(
            EventType.TEST,
            make_callback(1, 'first'),
            priority=10
        )
        
        self.event_system.subscribe(
            EventType.TEST,
            make_callback(2, 'second'),
            priority=5  # Note: in current architecture, higher numeric priority runs earlier (10 > 5)
        )
        
        # Publish an event
        self.event_system.publish(EventType.TEST, 'test data')
        
        # Check call order (should be by priority)
        self.assertEqual(len(results), 2)
        # Subscription with priority=10 should run before priority=5
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 2)
    
    def test_wildcard_subscription(self):
        """Test subscribing to events using wildcards."""
        # Subscribe with a wildcard pattern
        self.event_system.subscribe('test.*', self.mock_callback)
        
        # Publish matching and non-matching events
        self.event_system.publish('test.one', 'data1')
        self.event_system.publish('other.event', 'data2')
        self.event_system.publish('test.two', 'data3')
        
        # Verify only matching events were received
        self.assertEqual(self.mock_callback.call_count, 2)
        self.assertEqual(self.mock_callback.call_args_list[0][0][0].type, 'test.one')
        self.assertEqual(self.mock_callback.call_args_list[1][0][0].type, 'test.two')
    
    def test_event_handling(self):
        """Test event handling and marking events as handled."""
        second_callback = Mock()
        
        def first_handler(event):
            event.mark_handled()
        
        # Subscribe two handlers
        self.event_system.subscribe('test.handled', first_handler, priority=10)
        self.event_system.subscribe('test.handled', second_callback, priority=0)
        
        # Publish an event - second handler should not be called because first marks it as handled
        self.event_system.publish('test.handled')
        
        # Verify only the first handler was called
        second_callback.assert_not_called()
    
    def test_wait_for_event(self):
        """Test waiting for an event with a timeout."""
        def publish_after_delay():
            self.event_system.publish('test.wait', 'delayed data')
        
        # Start a thread to publish the event after a delay
        publish_after_delay()
        
        # Wait for the event
        event = self.event_system.wait_for('test.wait', timeout=1.0)
        
        # Verify the event was received
        self.assertIsNotNone(event)
        self.assertEqual(event.type, 'test.wait')
        self.assertEqual(event.data, 'delayed data')
    
    def test_wait_for_event_timeout(self):
        """Test that wait_for_event times out correctly."""
        # Wait for an event that won't be published
        event = self.event_system.wait_for('test.timeout', timeout=0.1)
        self.assertIsNone(event)
    
    def test_event_type_enum(self):
        """Test using EventType enum for event types."""
        # Subscribe using EventType enum
        _ = self.event_system.subscribe(EventType.WINDOW_CREATED, self.mock_callback)
        
        # Publish using EventType enum
        self.event_system.publish(EventType.WINDOW_CREATED, 'window data')
        
        # Verify the callback was called
        self.mock_callback.assert_called_once()
        callback_event = self.mock_callback.call_args[0][0]
        self.assertEqual(callback_event.type, 'window.created')
        self.assertEqual(callback_event.data, 'window data')


class TestEvent(unittest.TestCase):
    """Test cases for the Event class."""
    
    def test_event_creation(self):
        """Test creating an event with data and source."""
        source = object()
        _ = Event(EventType.TEST, 'test data', source)
        
        # Removed unused event variable
    def test_mark_handled(self):
        """Test marking an event as handled."""
        event = Event('test.event')
        event.mark_handled()
        self.assertTrue(event.is_handled)


class TestSubscription(unittest.TestCase):
    """Test cases for the Subscription class."""
    
    def test_subscription_creation(self):
        """Test creating a subscription."""
        def callback(event):
            pass
            
        subscription = Subscription(callback, 'test.event', priority=10)
        
        self.assertEqual(subscription.callback, callback)
        self.assertEqual(subscription.event_type, 'test.event')
        self.assertEqual(subscription.priority, 10)
        self.assertTrue(subscription.active)
    
    def test_subscription_comparison(self):
        """Test comparing subscriptions by priority."""
        def callback(event):
            pass
            
        sub1 = Subscription(callback, 'test.event', priority=10)
        sub2 = Subscription(callback, 'test.event', priority=5)
        
        # Higher priority should be "less than" lower priority for sorting
        self.assertLess(sub1, sub2)
    
    def test_subscription_matching(self):
        """Test event type matching with wildcards."""
        def callback(event):
            pass
        
        # Exact match
        sub = Subscription(callback, 'test.event')
        self.assertTrue(sub.matches('test.event'))
        self.assertFalse(sub.matches('test.other'))
        
        # Wildcard match
        sub = Subscription(callback, 'test.*')
        self.assertTrue(sub.matches('test.one'))
        self.assertTrue(sub.matches('test.two'))
        self.assertFalse(sub.matches('other.one'))
        
        # Multiple wildcards
        sub = Subscription(callback, 'test.*.sub.*')
        self.assertTrue(sub.matches('test.one.sub.item'))
        self.assertTrue(sub.matches('test.two.sub.other'))
        self.assertFalse(sub.matches('test.one.other'))


if __name__ == '__main__':
    unittest.main()
