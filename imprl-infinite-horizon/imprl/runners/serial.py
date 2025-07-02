def training_rollout(env, agent):

    terminated, truncated = False, False
    obs, info = env.reset()
    if hasattr(agent, "collect_state_info"):
        state = info["state"]
    agent.reset_episode()

    while not truncated and not terminated:

        # select action
        # _args are additional values (such as action_prob) to be stored
        # in the replay buffer
        action, *_args = agent.select_action(obs, training=True)

        # step in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.process_avoided_losses(info)

        # store experience in replay buffer
        if hasattr(agent, "collect_state_info"):
            next_state = info["state"]
            agent.process_experience(
                obs, state, *_args, next_obs, next_state, reward, terminated, truncated
            )
            # overwrite state
            state = next_state
        else:
            agent.process_experience(
                obs, *_args, next_obs, reward, terminated, truncated
            )

        # overwrite obs
        obs = next_obs

    # cal = agent._res / (agent._res + agent._res_loss)
    # agent.logger["episode_cal"] = cal

    return agent.episode_return


def evaluate_agent(env, agent):

    terminated, truncated = False, False
    obs, info = env.reset()
    agent.reset_episode(training=False)

    while not truncated and not terminated:

        # select action
        action = agent.select_action(obs, training=False)

        # step in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # process rewards
        agent.process_rewards(reward)
        agent.process_avoided_losses(info)

        # overwrite obs
        obs = next_obs

    cal = agent._res / (agent._res + agent._res_loss)

    return agent.episode_return, cal


def evaluate_heuristic(env, heuristic):

    done = False
    _, info = env.reset()
    obs = info["observation"]
    episode_return = 0
    time = 0

    while not done:

        # select action
        action = heuristic.policy(obs)

        # step in the environment
        _, reward, done, info = env.step(action)

        episode_return += reward

        # overwrite obs
        obs = info["observation"]
        time += 1

    # Total life cycle cost
    return episode_return