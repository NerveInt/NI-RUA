"""Optional external perception (object table, scene3d, VGGT height, held objects, pose planning)."""


def run_embodiment(runtime):
    """``runtime.embodiment`` (``Harness.embodiment``, the run's ``EmbodimentProfile`` resolved once from the
    backend) or None for a runtime without one (offline replays, unit-test doubles, a bare ``Config``)."""
    from ...robot.model.embodiments import EmbodimentProfile
    profile = getattr(runtime, "embodiment", None)
    return profile if isinstance(profile, EmbodimentProfile) else None


def embodiment_of(runtime):
    """The ``EmbodimentProfile`` every perception output is computed for (design 13 section 5.3): the run's
    (``run_embodiment``), else the default robot ``embodiments.ARX_X5_PROFILE``, as for a backend without an
    ``embodiment_profile`` hook. Example: ``embodiment_of(harness).gripper.full_open_aperture_m``."""
    profile = run_embodiment(runtime)
    if profile is not None:
        return profile
    from ...robot.model import embodiments
    return embodiments.ARX_X5_PROFILE
