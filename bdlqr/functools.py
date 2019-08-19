import inspect
from functools import partial


def getname(func):
    return (func.__name__ if hasattr(func, '__name__')
            else getname(func.__wrapped__) if hasattr(func, '__wrapped__')
            else getname(func.func) if hasattr(func, 'func')
            else NotImplemented())


def getdefaultkw(func, name):
    """
    Return default kw value
    """
    return inspect.signature(func).parameters[name].default
