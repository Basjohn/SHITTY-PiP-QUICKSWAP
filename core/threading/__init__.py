"""
Thread management utilities.

This package provides thread pooling and task management for concurrent operations.
"""
from typing import Any, Callable, Optional, Dict

# Back-compat exports (will be deprecated once canonical models are centralized)
from .task import Task  # noqa: F401
from .priority import TaskPriority  # noqa: F401
from .manager import ThreadManager, ThreadPoolType

# Create a singleton instance
_thread_manager: Optional[ThreadManager] = None

def get_thread_manager(config: Optional[Dict[ThreadPoolType, int]] = None) -> ThreadManager:
    """Get the singleton instance of the thread manager.
    
    Args:
        config: Optional pool configuration mapping ThreadPoolType -> max_workers
        
    Returns:
        ThreadManager: The singleton instance
    """
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager(config=config)
    return _thread_manager

def submit_task(pool_type: ThreadPoolType, func: Callable, *args, **kwargs) -> str:
    """Submit a task for execution in the thread pool.
    
    Args:
        pool_type: Which thread pool to use
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        str: Task ID that can be used to track or cancel the task
    """
    return get_thread_manager().submit_task(pool_type, func, *args, **kwargs)

def get_task_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """Get the result of a submitted task.
    
    Args:
        task_id: ID of the task to get results for
        timeout: Maximum time to wait for the result (None = wait forever)
        
    Returns:
        The result of the task execution
        
    Raises:
        KeyError: If no task with the given ID exists
        TimeoutError: If the timeout is reached before the task completes
        Exception: If the task raised an exception
    """
    return get_thread_manager().get_task_result(task_id, timeout)

def cancel_task(task_id: str) -> bool:
    """Cancel a running task.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        bool: True if the task was successfully cancelled, False otherwise
    """
    return get_thread_manager().cancel_task(task_id)

__all__ = [
    'ThreadManager',
    'Task',
    'TaskPriority',
    'get_thread_manager',
    'submit_task',
    'get_task_result',
    'cancel_task',
]
