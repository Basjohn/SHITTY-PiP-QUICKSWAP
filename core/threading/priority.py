"""
Task priority definitions for thread management.

This module defines the priority levels for tasks in the thread pool.
"""
from enum import Enum


class TaskPriority(Enum):
    """Priority levels for tasks in the thread pool.
    
    Higher priority values will be executed before lower priority values.
    """
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
