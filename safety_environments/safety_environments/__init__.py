import gymnasium

MINIGRID_ENVS = [
    "LavaGapS5-Relaxed",
    "LavaGapS5",
    "LavaGapS7-Relaxed",
    "LavaGapS7",
    "DistShift1",
    "DistShift1-Relaxed",
    "DistShift2",
    "DistShift2-Relaxed",
    "LavaCrossingS9N1-Relaxed",
    "LavaCrossingS9N2-Relaxed",
    "LavaCrossingS9N3-Relaxed",
    "LavaCrossingS11N5-Relaxed",
]

gymnasium.register(
    "LavaGapS5-Relaxed",
    entry_point="safety_environments.envs:LavaGapMinigrid",
    kwargs={
        "grid_size": 5,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaGapS5",
    entry_point="safety_environments.envs:LavaGapMinigrid",
    kwargs={
        "grid_size": 5,
        "lava_death": True,
        "test": False,
    },
)

gymnasium.register(
    "LavaGapS7-Relaxed",
    entry_point="safety_environments.envs:LavaGapMinigrid",
    kwargs={
        "grid_size": 7,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaGapS7",
    entry_point="safety_environments.envs:LavaGapMinigrid",
    kwargs={
        "grid_size": 7,
        "lava_death": True,
        "test": False,
    },
)

gymnasium.register(
    "DistShift1",
    entry_point="safety_environments.envs:DistShiftMinigrid",
    kwargs={
        "version": 1,
        "lava_death": True,
        "test": False,
    },
)

gymnasium.register(
    "DistShift1-Relaxed",
    entry_point="safety_environments.envs:DistShiftMinigrid",
    kwargs={
        "version": 1,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "DistShift2",
    entry_point="safety_environments.envs:DistShiftMinigrid",
    kwargs={
        "version": 2,
        "lava_death": True,
        "test": False,
    },
)

gymnasium.register(
    "DistShift2-Relaxed",
    entry_point="safety_environments.envs:DistShiftMinigrid",
    kwargs={
        "version": 2,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaCrossingS9N1-Relaxed",
    entry_point="safety_environments.envs:CrossingMinigrid",
    kwargs={
        "grid_size": 9,
        "num_crossing": 1,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaCrossingS9N2-Relaxed",
    entry_point="safety_environments.envs:CrossingMinigrid",
    kwargs={
        "grid_size": 9,
        "num_crossing": 2,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaCrossingS9N3-Relaxed",
    entry_point="safety_environments.envs:CrossingMinigrid",
    kwargs={
        "grid_size": 9,
        "num_crossing": 3,
        "lava_death": False,
        "test": False,
    },
)

gymnasium.register(
    "LavaCrossingS11N5-Relaxed",
    entry_point="safety_environments.envs:CrossingMinigrid",
    kwargs={
        "grid_size": 11,
        "num_crossing": 6,
        "lava_death": False,
        "test": False,
    },
)
