# import pytest

# import imprl.agents
# from imprl.runners.serial import training_rollout
# from imprl.agents.configs.get_config import load_config


# def dummy_training_rollouts(algorithm, env):

#     config = load_config(algorithm=algorithm)
#     agent_class = imprl.agents.get_agent_class(algorithm)
#     agent = agent_class(env, config, "cpu")

#     # training loop: 100 episodes
#     for _ in range(50):
#         training_rollout(env, agent)


# @pytest.mark.parametrize(
#     "env_name, env_setting",
#     [
#         ("k_out_of_n", "hard-5-of-5"),
#         ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
#         ("matrix_game", "climb_game"),
#     ],
# )
# @pytest.mark.parametrize("algorithm", ["DDQN", "JAC"])
# def test_single_agent(env_name, env_setting, algorithm):

#     env = imprl.envs.make(
#         env_name, env_setting, single_agent=True, percept_type="belief"
#     )

#     dummy_training_rollouts(algorithm, env)


# @pytest.mark.parametrize(
#     "env_name, env_setting",
#     [
#         ("k_out_of_n", "hard-5-of-5"),
#         ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
#         ("matrix_game", "climb_game"),
#     ],
# )
# @pytest.mark.parametrize(
#     "algorithm",
#     ["DCMAC", "DDMAC", "IACC", "IACC_PS", "VDN_PS", "QMIX_PS", "IAC", "IAC_PS"],
# )
# def test_multi_agent(env_name, env_setting, algorithm):

#     env = imprl.envs.make(
#         env_name, env_setting, single_agent=False, percept_type="belief"
#     )

#     dummy_training_rollouts(algorithm, env)
