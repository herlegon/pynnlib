# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import copy
import inspect
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional, Type
import warnings


_lock = threading.RLock()

def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()




class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain the ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, f'{cls} must have the `name` argument'
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = '', **kwargs):
        assert isinstance(name, str) and name, \
            'name argument must be an non-empty string.'
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), \
            f'type of name should be str, but got {type(cls)}'
        instance_dict = cls._instance_dict  # type: ignore
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)  # type: ignore
            instance_dict[name] = instance  # type: ignore
        elif kwargs:
            warnings.warn(
                f'{cls} instance named of {name} has been created, '
                'the method `get_instance` should not accept any other '
                'arguments')
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f'Before calling {cls.__name__}.get_current_instance(), you '
                'should call get_instance(name=xxx) at least once.')
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name


class DefaultScope(ManagerMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Consider the case of resetting the current ``Registry`` by
    ``default_scope`` in the internal module which cannot access runner
    directly, it is difficult to get the ``default_scope`` defined in
    ``Runner``. However, if ``Runner`` created ``DefaultScope`` instance
    by given ``default_scope``, the internal module can get
    ``default_scope`` by ``DefaultScope.get_current_instance`` everywhere.

    Args:
        name (str): Name of default scope for global access.
        scope_name (str): Scope of current task.

    Examples:
        >>> from mmengine.model import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('task', scope_name='mmdet')
        >>> # Get default scope globally.
        >>> scope_name = DefaultScope.get_instance('task').scope_name
    """

    def __init__(self, name: str, scope_name: str):
        super().__init__(name)
        assert isinstance(
            scope_name,
            str), (f'scope_name should be a string, but got {scope_name}')
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> Optional['DefaultScope']:
        """Get latest created default scope.

        Since default_scope is an optional argument for ``Registry.build``.
        ``get_current_instance`` should return ``None`` if there is no
        ``DefaultScope`` created.

        Examples:
            >>> default_scope = DefaultScope.get_current_instance()
            >>> # There is no `DefaultScope` created yet,
            >>> # `get_current_instance` return `None`.
            >>> default_scope = DefaultScope.get_instance(
            >>>     'instance_name', scope_name='mmengine')
            >>> default_scope.scope_name
            mmengine
            >>> default_scope = DefaultScope.get_current_instance()
            >>> default_scope.scope_name
            mmengine

        Returns:
            Optional[DefaultScope]: Return None If there has not been
            ``DefaultScope`` instance created yet, otherwise return the
            latest created DefaultScope instance.
        """
        _accquire_lock()
        if cls._instance_dict:
            instance = super().get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: Optional[str]) -> Generator:
        """Overwrite the current default scope with `scope_name`"""
        if scope_name is None:
            yield
        else:
            tmp = copy.deepcopy(cls._instance_dict)
            # To avoid create an instance with the same name.
            time.sleep(1e-6)
            cls.get_instance(f'overwrite-{time.time()}', scope_name=scope_name)
            try:
                yield
            finally:
                cls._instance_dict = tmp
