import os
import yaml


def make(name, setting, single_agent=False, **env_kwargs):

    if name in [
        "k_out_of_n",
        "k_out_of_n_v3",
        "k_out_of_n_infinite",
        "k_out_of_n_shaping",
    ]:

        import imprl.envs.structural_envs.k_out_of_n
        import imprl.envs.structural_envs.k_out_of_n_v3
        import imprl.envs.structural_envs.k_out_of_n_infinite
        from imprl.envs.structural_envs.single_agent_wrapper import SingleAgentWrapper
        from imprl.envs.structural_envs.multi_agent_wrapper import MultiAgentWrapper

        # get class KOutOfN
        module = getattr(imprl.envs.structural_envs, name)
        env_class = getattr(module, "KOutOfN")

        rel_path_config = f"structural_envs/env_configs/{setting}.yaml"
        rel_path_baselines = f"structural_envs/baselines.yaml"

    elif name == "matrix_game":

        import imprl.envs.game_envs.matrix_game
        from imprl.envs.game_envs.single_agent_wrapper import SingleAgentWrapper
        from imprl.envs.game_envs.multi_agent_wrapper import MultiAgentWrapper

        # get class MatrixGame
        module = getattr(imprl.envs.game_envs, name)
        env_class = getattr(module, "MatrixGame")

        rel_path_config = f"game_envs/env_configs/{setting}.yaml"
        rel_path_baselines = f"game_envs/baselines.yaml"

    # get the environment config
    pwd = os.path.dirname(__file__)
    env_config_path = os.path.join(pwd, rel_path_config)
    baselines_path = os.path.join(pwd, rel_path_baselines)

    with open(env_config_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(baselines_path) as file:
        all_baselines = yaml.load(file, Loader=yaml.FullLoader)

    baselines = all_baselines[name][setting]

    # create the environment
    env = env_class(env_config, baselines, **env_kwargs)

    # wrap the environment
    if single_agent:
        env = SingleAgentWrapper(env)
    else:
        env = MultiAgentWrapper(env)

    return env
