# Copyright (c) OpenMMLab. All rights reserved.
from functools import wraps
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import warnings

from .config import Config, ConfigDict
from .default_scope import ManagerMixin
from .registry import Registry

if TYPE_CHECKING:
    import torch.nn as nn
    # from mmengine.runner import Runner


class _ParamScheduler:
    """Base class for parameter schedulers.

    It should be inherited by all schedulers that schedule parameters in the
    optimizer's ``param_groups``. All subclasses should overwrite the
    ``_get_value()`` according to their own schedule strategy.
    The implementation is motivated by
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resuming without
            state dict. Default value ``-1`` means the ``step`` function is
            never be called before. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """  # noqa: E501

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, BaseOptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)  # type: ignore
        self.optimizer._global_step = -1  # type: ignore

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print(
                f'Adjusting parameter value of group {group} to {value:.4e}.')

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]


def build_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """
    # Avoid circular import

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
            f'but got {type(default_args)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.scope}::{registry.name} registry. '  # noqa: E501
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at '
                    'https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        # If `obj_cls` inherits from `ManagerMixin`, it should be
        # instantiated by `ManagerMixin.get_instance` to ensure that it
        # can be accessed globally.
        if inspect.isclass(obj_cls) and \
                issubclass(obj_cls, ManagerMixin):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args)  # type: ignore

        if (inspect.isclass(obj_cls) or inspect.isfunction(obj_cls)
                or inspect.ismethod(obj_cls)):
            print_log(
                f'An `{obj_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
                'registry, and its implementation can be found in '
                f'{obj_cls.__module__}',  # type: ignore
                logger='current',
                level=logging.DEBUG)
        else:
            print_log(
                'An instance is built from registry, and its constructor '
                f'is {obj_cls}',
                logger='current',
                level=logging.DEBUG)
        return obj


def build_runner_from_cfg(cfg: Union[dict, ConfigDict, Config],
                          registry: Registry): # -> 'Runner':
    """Build a Runner object.

    Examples:
        >>> from mmengine.registry import Registry, build_runner_from_cfg
        >>> RUNNERS = Registry('runners', build_func=build_runner_from_cfg)
        >>> @RUNNERS.register_module()
        >>> class CustomRunner(Runner):
        >>>     def setup_env(env_cfg):
        >>>         pass
        >>> cfg = dict(runner_type='CustomRunner', ...)
        >>> custom_runner = RUNNERS.build(cfg)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. If "runner_type" key
            exists, it will be used to build a custom runner. Otherwise, it
            will be used to build a default runner.
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed runner object.
    """
    from .config import Config, ConfigDict

    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    # Runner should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.get('runner_type', 'Runner')
        if isinstance(obj_type, str):
            runner_cls = registry.get(obj_type)
            if runner_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                )
        elif inspect.isclass(obj_type):
            runner_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        runner = runner_cls.from_cfg(args)  # type: ignore
        print(
            f'An `{runner_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
            'registry, its implementation can be found in'
            f'{runner_cls.__module__}')
        return runner


def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
) -> 'nn.Module':
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    # from .model import Sequential
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_optimizer_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    if 'type' in cfg  and 'Adafactor' == cfg['type']:
            # and digit_version(torch.__version__) >= digit_version('2.5.0'):
        print(
            'the torch version of Adafactor is registered as TorchAdafactor')
    return build_from_cfg(cfg, registry, default_args)


def build_scheduler_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None
) -> '_ParamScheduler':
    """Builds a ``ParamScheduler`` instance from config.

    ``ParamScheduler`` supports building instance by its constructor or
    method ``build_iter_from_epoch``. Therefore, its registry needs a build
    function to handle both cases.

    Args:
        cfg (dict or ConfigDict or Config): Config dictionary. If it contains
            the key ``convert_to_iter_based``, instance will be built by method
            ``convert_to_iter_based``, otherwise instance will be built by its
            constructor.
        registry (:obj:`Registry`): The ``PARAM_SCHEDULERS`` registry.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. It must contain key ``optimizer``. If
            ``convert_to_iter_based`` is defined in ``cfg``, it must
            additionally contain key ``epoch_length``. Defaults to None.

    Returns:
        object: The constructed ``ParamScheduler``.
    """
    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        convert_to_iter = args.pop('convert_to_iter_based', False)
        if convert_to_iter:
            scheduler_type = args.pop('type')
            assert 'epoch_length' in args and args.get('by_epoch', True), (
                'Only epoch-based parameter scheduler can be converted to '
                'iter-based, and `epoch_length` should be set')
            if isinstance(scheduler_type, str):
                scheduler_cls = registry.get(scheduler_type)
                if scheduler_cls is None:
                    raise KeyError(
                        f'{scheduler_type} is not in the {registry.name} '
                        'registry. Please check whether the value of '
                        f'`{scheduler_type}` is correct or it was registered '
                        'as expected. More details can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                    )
            elif inspect.isclass(scheduler_type):
                scheduler_cls = scheduler_type
            else:
                raise TypeError('type must be a str or valid type, but got '
                                f'{type(scheduler_type)}')
            return scheduler_cls.build_iter_from_epoch(  # type: ignore
                **args)
        else:
            args.pop('epoch_length', None)
            return build_from_cfg(args, registry)
