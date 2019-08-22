import inspect
from functools import partial


def getname(func):
    return (func.__name__ if hasattr(func, '__name__')
            else getname(func.__wrapped__) if hasattr(func, '__wrapped__')
            else getname(func.func) if hasattr(func, 'func')
            else NotImplemented())


def _getdefaultkw1(func, name):
    return inspect.signature(func).parameters[name].default


def _getdefaultkw(func, namepath, sep):
    name_path_list = namepath.split(sep)
    val = _getdefaultkw1(func, name_path_list[0])
    if len(name_path_list) > 1:
        func2 = val
        return getdefaultkw(func2, name_path_list[1:])
    else:
        return val


def getdefaultkw(func, *namepaths, sep="."):
    """
    Return default kw value
    """
    return [_getdefaultkw(func, np, sep=sep) for np in namepaths]


def list_extend(L, iterable=[]):
    return L + list(iterable)


def list_extendable(L):
    return partial(list_extend, L)


def dict_extendable(D):
    return partial(dict, D)
