"""Explicit environment construction; injected backends use the same Harness port."""
from types import MappingProxyType


def _robodojo(**kwargs):
    from .robodojo.robodojo import RobotBackend
    return RobotBackend(**kwargs)


def _so101_real(**kwargs):
    from .so101.backend import RobotBackend
    return RobotBackend(**kwargs)


def _libero(**kwargs):
    from .libero.backend import RobotBackend
    return RobotBackend(**kwargs)


ENVIRONMENT_BACKENDS = MappingProxyType({
    "robodojo": _robodojo,
    "so101_real": _so101_real,
    "libero": _libero,
})


def create_backend(*, config, **kwargs):
    """Resolve an environment first; its adapter validates the robot binding."""
    try:
        factory = ENVIRONMENT_BACKENDS[config.backend.environment]
    except KeyError:
        raise ValueError(f"unsupported_environment: {config.backend.environment}") from None
    return factory(config=config, **kwargs)
