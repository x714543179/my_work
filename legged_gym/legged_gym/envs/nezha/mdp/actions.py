"""Action processing for the Nezha spring jump task."""


def command_latency(env, actions, enabled=True, randomize=True, latency_range=(1, 3)):
    if not enabled:
        return actions.clone()

    max_latency = int(latency_range[1])
    env.cmd_action_latency_buffer[:, :, 1:] = env.cmd_action_latency_buffer[
        :, :, :max_latency
    ].clone()
    env.cmd_action_latency_buffer[:, :, 0] = actions
    return env.cmd_action_latency_buffer[
        env._env_indices, :, env.cmd_action_latency_simstep.long()
    ]
