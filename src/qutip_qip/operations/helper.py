"""
This should be moved to utils.py after deprecated functions there are removed in the next version
"""

from functools import lru_cache, partial


class class_or_instance_method:
    """Bind to the class when accessed on a class value (X.inverse()),
    to the instance otherwise (RX(0.5).inverse()). Used here and in Controlled."""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return partial(self.f, owner if obj is None else obj)


class cached_class_or_instance_method(class_or_instance_method):
    def __set_name__(self, owner, name):
        self.slot = f"_{name}_cache"

    def __get__(self, obj, owner):
        if self.slot not in vars(owner):
            setattr(owner, self.slot, lru_cache(maxsize=128)(self.f))
        return partial(vars(owner)[self.slot], owner if obj is None else obj)
