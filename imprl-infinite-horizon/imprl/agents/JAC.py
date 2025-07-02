import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork


class JointActorCritic(PGAgent):
    name = "JAC"
    full_name = "Joint Actor-Critic"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        assert (
            env.__class__.__name__ == "SingleAgentWrapper"
        ), "JAC only supports single-agent environments"

        ## Neural Networks
        n_inputs = self.env.perception_dim
        n_outputs_actor = self.env.action_dim
        n_outputs_critic = 1

        self.actor_config["architecture"] = (
            [n_inputs] + self.actor_config["hidden_layers"] + [n_outputs_actor]
        )
        self.critic_config["architecture"] = (
            [n_inputs] + self.critic_config["hidden_layers"] + [n_outputs_critic]
        )

        # Actors (info centralised: can observe the entire system state/belief)
        self.actor = ActorNetwork(
            self.actor_config["architecture"],
            initialization="orthogonal",
            optimizer=self.actor_config["optimizer"],
            learning_rate=self.actor_config["lr"],
            lr_scheduler=self.actor_config["lr_scheduler"],
        ).to(device)

        # Critic (info centralised: can observe entire system state/belief)
        self.critic = NeuralNetwork(
            self.critic_config["architecture"],
            initialization="orthogonal",
            optimizer=self.critic_config["optimizer"],
            learning_rate=self.critic_config["lr"],
            lr_scheduler=self.critic_config["lr_scheduler"],
        ).to(device)

    def get_random_action(self):

        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)
        action_prob = 1 / self.env.action_dim

        return action, t_action, action_prob

    def get_greedy_action(self, observation, training):

        # get action from actor network
        flat_obs = self.env.system_percept(observation)
        t_observation = torch.tensor(flat_obs).to(self.device)
        action_dist = self.actor.forward(t_observation, training)
        t_action = action_dist.sample()  # index of joint action

        action = t_action.cpu().detach().numpy()

        if training:
            log_prob = action_dist.log_prob(t_action)
            action_prob = torch.exp(log_prob)
            return action, t_action, action_prob
        else:
            return action

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

    def compute_log_prob(self, t_beliefs, t_idx_actions):

        # shape: (batch_size, n_actions)
        action_dists = self.actor.forward(t_beliefs)

        # compute log prob of each action under current policy
        # shape: (batch_size)
        _log_probs = action_dists.log_prob(t_idx_actions)

        return _log_probs

    def compute_sample_weight(self, log_probs, t_action_probs):

        new_probs = torch.exp(log_probs)

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
