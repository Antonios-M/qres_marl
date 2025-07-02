import numpy as np
import torch
import gymnasium as gym

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork


class IndependentActorCentralisedCriticParameterSharing(PGAgent):
    name = "IACC-PS"
    full_name = "Independent Actor Centralised Critic with Parameter Sharing"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.n_agent_actions = [space.n for space in env.action_space]
        self.n_agents = len(self.n_agent_actions)

        ## Neural networks
        # assume homogeneous local observations
        # shape: (local_obs+shared_obs) + n_agents (id)
        state_dim = gym.spaces.utils.flatdim(env.perception_space)

        obs, info = env.reset()
        ma_idx_obs = env.multiagent_idx_percept(obs)
        n_inputs_actor = ma_idx_obs.shape[1]

        n_inputs_critic = state_dim
        n_outputs_actor = self.n_agent_actions[0]
        n_outputs_critic = 1

        self.actor_config["architecture"] = (
            [n_inputs_actor] + self.actor_config["hidden_layers"] + [n_outputs_actor]
        )
        self.critic_config["architecture"] = (
            [n_inputs_critic] + self.critic_config["hidden_layers"] + [n_outputs_critic]
        )

        # Actor
        # (decentralised: can only observe component state/belief)
        # but parameters are shared
        # actions for individual component
        self.actor = ActorNetwork(
            self.actor_config["architecture"],
            initialization="orthogonal",
            optimizer=self.actor_config["optimizer"],
            learning_rate=self.actor_config["lr"],
            lr_scheduler=self.actor_config["lr_scheduler"],
        ).to(device)

        # Critic (centralised: can observe entire system state)
        self.critic = NeuralNetwork(
            self.critic_config["architecture"],
            initialization="orthogonal",
            optimizer=self.critic_config["optimizer"],
            learning_rate=self.critic_config["lr"],
            lr_scheduler=self.critic_config["lr_scheduler"],
        ).to(device)

    def get_greedy_action(self, observation, training):

        # convert to tensor
        ma_idx_obs = self.env.multiagent_idx_percept(observation)
        t_ma_obs = torch.tensor(ma_idx_obs).to(self.device)

        action_dist = self.actor.forward(t_ma_obs, training)
        t_action = action_dist.sample()
        action = t_action.cpu().detach().numpy()

        if training:
            log_prob = action_dist.log_prob(t_action)
            action_prob = torch.prod(torch.exp(log_prob), dim=-1)  # joint action prob
            return action, t_action, action_prob
        else:
            return action

    def process_experience(
        self, belief, action, action_prob, next_belief, reward, terminated, truncated
    ):

        self.process_rewards(reward)

        self.replay_memory.store_experience(
            self.env.multiagent_idx_percept(belief),
            self.env.system_percept(belief),
            action,
            action_prob,
            self.env.system_percept(next_belief),
            reward,
            terminated,
            truncated,
        )

        # start batch learning once sufficient samples are available
        if self.total_time > 10 * self.batch_size:
            sample_batch = self.replay_memory.sample_batch(self.batch_size)
            self.train(*sample_batch)

        if terminated or truncated:
            self.logger["episode"] = self.episode
            self.logger["episode_cost"] = -self.episode_return

    def get_future_values(self, t_next_beliefs):

        # bootstrapping
        # shape: (batch_size, 1)
        future_values = self.critic.forward(t_next_beliefs)

        return future_values

    def compute_log_prob(self, t_ma_beliefs, t_actions):

        # get actions dists from each actor
        # logits shape: (batch_size, num_components, num_actions)
        action_dists = self.actor.forward(t_ma_beliefs)

        # compute log prob of each action under current policy
        # shape: (batch_size, num_components)
        _log_probs = action_dists.log_prob(t_actions)

        return _log_probs

    def compute_sample_weight(self, joint_log_probs, joint_action_probs):

        new_probs = torch.exp(joint_log_probs)

        # true dist / proposal dist
        weights = new_probs / joint_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        # shape: (batch_size, 1)
        return weights.detach().reshape(-1, 1)

    def compute_loss(self, *args):

        # preprocess inputs
        (
            t_ma_beliefs,
            t_beliefs,
            t_actions,
            t_action_probs,
            t_next_beliefs,
            t_rewards,
            t_terminations,
            t_truncations,
        ) = self._preprocess_inputs(*args)

        # Value function update
        current_values = self.critic.forward(t_beliefs)  # shape: (batch_size, 1)
        td_targets = self.compute_td_target(
            t_next_beliefs, t_rewards, t_terminations, t_truncations
        )  # shape: (batch_size, 1)

        advantage = (td_targets - current_values).detach()  # shape: (batch_size, 1)

        # compute log_prob actions
        # shape: (batch_size, num_components)
        t_log_probs = self.compute_log_prob(t_ma_beliefs, t_actions)

        # compute joint probs
        # sum over all actions for each component
        # shape: (batch_size)
        t_joint_log_probs = torch.sum(t_log_probs, dim=-1)

        # compute importance sampling weights
        # shape: (batch_size)
        weights = self.compute_sample_weight(t_joint_log_probs.detach(), t_action_probs)

        # TD_target = r_t + γ V(b')
        # L_V(theta) = E[w (TD_target - V(b))^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        # t_log_probs @ advantage
        # (B, M) * (B, 1) => (B, M)
        # torch.sum(B, M, dim=1,keepdim=True) => (B, 1)
        # torch.mean(-(B, 1) * (B, 1)) => scalar
        actor_loss = torch.mean(
            -torch.sum(t_log_probs * advantage, dim=1, keepdim=True) * weights
        )

        return actor_loss, critic_loss

    def _preprocess_inputs(
        self,
        ma_idx_beliefs,
        sytem_beliefs,
        actions,
        action_probs,
        system_next_beliefs,
        rewards,
        terminations,
        truncations,
    ):
        t_beliefs = torch.tensor(np.array(sytem_beliefs)).to(self.device)
        t_next_beliefs = torch.tensor(np.array(system_next_beliefs)).to(self.device)
        t_ma_beliefs = torch.tensor(np.array(ma_idx_beliefs)).to(self.device)

        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)

        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)

        t_terminations = (
            torch.tensor(terminations, dtype=torch.int).reshape(-1, 1).to(self.device)
        )
        t_truncations = (
            torch.tensor(truncations, dtype=torch.int).reshape(-1, 1).to(self.device)
        )


        return (
            t_ma_beliefs,
            t_beliefs,
            t_actions,
            t_action_probs,
            t_next_beliefs,
            t_rewards,
            t_terminations,
            t_truncations,
        )

    def save_weights(self, path, episode):
        torch.save(self.actor.state_dict(), f"{path}/actor_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{episode}.pth")

    def load_weights(self, path, episode):

        # load actor weights
        full_path = f"{path}/actor_{episode}.pth"
        self.actor.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )

        # load critic weights
        full_path = f"{path}/critic_{episode}.pth"
        self.critic.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )
