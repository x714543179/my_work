"""Termination terms for Nezha spring jumping."""


def base_height_below(env, minimum_height=0.2):
    return env.root_states[:, 2] <= minimum_height
