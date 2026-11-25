"""SO-101 real-hardware backend (environment ``so101_real``, robot ``so101``).

``backend`` (parent process), ``worker`` (so101 interpreter, owns bus and
cameras), ``driver`` (LeRobot/OpenCV seams and test fakes), ``contract`` (pure
compile/validate) and ``deployment`` (validated machine/threshold file).
"""
