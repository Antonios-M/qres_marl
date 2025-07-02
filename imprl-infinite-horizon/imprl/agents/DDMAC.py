import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.primitives.MultiAgentActors import MultiAgentActors


class DeepDecentralisedMultiAgentActorCritic(PGAgent):
    name = "DDMAC"
    full_name = "Deep Decentralised Multi-Agent Actor-Critic"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        assert env.single_agent == False, "DDMAC only supports multi-agent environments"

        ## Neural networks
        obs, info = env.reset()
        ma_system_obs = env.system_percept(obs)
        n_inputs = ma_system_obs.shape[-1]

        self.n_agent_actions = [space.n for space in env.action_space]
        self.n_agents = len(self.n_agent_actions)

        n_outputs_actor = self.n_agent_actions[0]
        n_outputs_critic = 1

        self.actor_config["architecture"] = (
            [n_inputs] + self.actor_config["hidden_layers"] + [n_outputs_actor]
        )
        self.critic_config["architecture"] = (
            [n_inputs] + self.critic_config["hidden_layers"] + [n_outputs_critic]
        )

        # Actors
        # (decentralised: can observe the entire system state/belief)
        # but unlike DCMAC, parameters are not shared
        # actions for individual component
        # currently only supports discrete+homogeneous action spaces
        self.actor = MultiAgentActors(
            self.n_agents, self.n_agent_actions[0], self.actor_config, device
        )

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
        system_obs = self.env.system_percept(observation)
        t_observation = torch.tensor(system_obs).to(self.device)

        return self.actor.forward(t_observation, training=training, ind_obs=False)

    def process_experience(
        self, belief, action, action_prob, next_belief, reward, terminated, truncated
    ):

        belief = self.env.system_percept(belief)
        next_belief = self.env.system_percept(next_belief)

        super().process_experience(
            belief, action, action_prob, next_belief, reward, terminated, truncated
        )

    def get_future_values(self, t_next_beliefs):

        # bootstrapping
        future_values = self.critic.forward(t_next_beliefs)

        return future_values

    def compute_log_prob(self, t_beliefs, t_actions):

        _log_probs = torch.ones((self.batch_size, self.n_agents)).to(self.device)

        # get actions from each actor network
        for k, actor_network in enumerate(self.actor.networks):
            action_dists = actor_network.forward(t_beliefs)

            # compute log prob of each action under current policy
            # shape: (batch_size)
            _log_probs[:, k] = action_dists.log_prob(t_actions[:, k])

        # compute joint probs
        # shape: (batch_size)
        joint_log_probs = torch.sum(_log_probs, dim=-1)

        return joint_log_probs

    def compute_sample_weight(self, joint_log_probs, t_action_probs):

        new_probs = torch.exp(joint_log_probs)

        # true dist / proposal dist
        weights = new_probs / t_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        return weights.detach()

    def compute_loss(self, *args):

        (
            t_beliefs,
            t_next_beliefs,
            t_actions,
            t_action_probs,
            t_rewards,
            t_terminations,
            t_truncations,
        ) = self._preprocess_inputs(*args)

        current_values = self.critic.forward(t_beliefs)
        td_targets = self.compute_td_target(
            t_next_beliefs, t_rewards, t_terminations, t_truncations
        )

        t_log_probs = self.compute_log_prob(t_beliefs, t_actions)

        weights = self.compute_sample_weight(t_log_probs.detach(), t_action_probs)

        # TD_target = r_t + γ V(b')
        # L_V(theta) = E[w (TD_target - V(b))^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        advantage = (td_targets - current_values).detach().flatten()

        # max  (log_prob * advantage)
        # min -(log_prob * advantage)
        actor_loss = torch.mean(-t_log_probs * advantage * weights)

        return actor_loss, critic_loss

    def save_weights(self, path, episode):

        for i, actor_network in enumerate(self.actor.networks):
            torch.save(actor_network.state_dict(), f"{path}/actor_{i+1}_{episode}.pth")

        torch.save(self.critic.state_dict(), f"{path}/critic_{episode}.pth")

    def load_weights(self, path, episode):

        # load actor weights
        for i, actor_network in enumerate(self.actor.networks):
            actor_network.load_state_dict(
                torch.load(
                    f"{path}/actor_{i+1}_{episode}.pth",
                    map_location=torch.device("cpu"),
                )
            )

        # load critic weights
        self.critic.load_state_dict(
            torch.load(f"{path}/critic_{episode}.pth", map_location=torch.device("cpu"))
        )
