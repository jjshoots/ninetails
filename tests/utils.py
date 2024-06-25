"""Utils used during testing."""

from typing import Any

import numpy as np


def all_equal(obj1: Any, obj2: Any) -> bool:
    """all_equal.

    Args:
        obj1 (Any): obj1
        obj2 (Any): obj2

    Returns:
        bool:
    """
    # if both objects are of different types, they can't be equivalent
    if type(obj1) != type(obj2):
        return False

    # Check based on type
    if isinstance(obj1, dict):
        # check if dictionaries have the same keys and equivalent values
        if len(obj1) != len(obj2):
            return False
        for key in obj1:
            if key not in obj2 or not all_equal(obj1[key], obj2[key]):
                return False
        return True

    elif isinstance(obj1, (list, tuple)):
        # check if lists and tuples have the same length and equivalent elements
        if len(obj1) != len(obj2):
            return False
        for i in range(len(obj1)):
            if not all_equal(obj1[i], obj2[i]):
                return False
        return True

    elif isinstance(obj1, np.ndarray):
        # check that np arrays have the same shape
        if obj1.shape != obj2.shape:
            return False
        if not np.all(obj1 == obj2):
            return False
        return True

    # for other types (int, str, float, etc.), check for direct equivalence
    return obj1 == obj2
