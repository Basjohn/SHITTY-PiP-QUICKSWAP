"""
Task module for thread management.

This module provides the Task class which represents an asynchronous task
that can be executed by the ThreadManager.
"""
import time
import uuid
from typing import Any, Callable, Optional

class Task:
    """Represents an asynchronous task that can be executed by the ThreadManager."""
    
    def __init__(self, func: Callable, args: tuple = None, kwargs: dict = None):
        """Initialize the task.
        
        Args:
            func: The function to execute
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
        """
        self.id = str(uuid.uuid4())
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.future = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.exception = None
        self._completed = False  # Lock-free: Use atomic bool instead of threading.Event
    
    def execute(self, executor) -> None:
        """Execute the task using the given executor.
        
        Args:
            executor: The executor to use for running the task
        """
        self.started_at = time.time()
        
        def _wrapper():
            try:
                self.result = self.func(*self.args, **self.kwargs)
                return self.result
            except Exception as e:
                self.exception = e
                raise
            finally:
                self.completed_at = time.time()
                self._completed = True  # Lock-free: Set completion flag atomically
        
        self.future = executor.submit(_wrapper)
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the task to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if task completed, False if timeout occurred
        """
        # Lock-free: Busy wait with yielding instead of blocking Event.wait()
        import time
        start_time = time.time()
        while not self._completed:
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            time.sleep(0.001)  # Small yield to prevent busy spinning
        return True
    
    def is_done(self) -> bool:
        """Check if the task is completed.
        
        Returns:
            True if task is completed
        """
        return self._completed  # Lock-free: Atomic read
    
    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Get the result of the task.
        
        Args:
            timeout: Maximum time to wait for the result
            
        Returns:
            The result of the task execution
            
        Raises:
            TimeoutError: If the timeout is reached before the task completes
            Exception: If the task raised an exception
        """
        if not self.wait(timeout):
            raise TimeoutError("Timed out waiting for task to complete")
        
        if self.exception is not None:
            raise self.exception
            
        return self.result
