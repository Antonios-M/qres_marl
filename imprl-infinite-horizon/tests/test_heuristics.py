import pytest

import imprl.envs
from imprl.baselines.failure_replace import FailureReplace
from imprl.baselines.do_nothing import DoNothing
from imprl.runners.serial import evaluate_agent


@pytest.mark.parametrize(
    "env_name, env_setting",
    [
        ("k_out_of_n", "hard-5-of-5"),
        ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
    ],
)
def test_failure_replace(env_name, env_setting):
    env = imprl.envs.make(
        env_name, env_setting, single_agent=False, percept_type="belief"
    )
    fr_agent = FailureReplace(env)

    # check if episode return is not None
    assert evaluate_agent(env, fr_agent) is not None

@pytest.mark.parametrize(
    "env_name, env_setting",
    [
        ("k_out_of_n", "hard-5-of-5"),
        ("k_out_of_n_infinite", "hard-1-of-4_infinite"),
    ],
)
def test_do_nothing(env_name, env_setting):
    env = imprl.envs.make(
        env_name, env_setting, single_agent=False, percept_type="belief"
    )

    dn_agent = DoNothing(env)

    # check if episode return is not None
    assert evaluate_agent(env, dn_agent) is not None
    