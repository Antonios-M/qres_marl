import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork


class IndependentActorCriticParameterSharing(PGAgent):
    name = "IAC-PS"
    full_name = "Independent Actor-Critic with Parameter Sharing"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.n_agent_actions = [space.n for space in env.action_space]
        self.n_agents = len(self.n_agent_actions)

        ## Neural networks
        obs, info = env.reset()
        ma_idx_obs = env.multiagent_idx_percept(obs)
        n_inputs = ma_idx_obs.shape[1]

        n_outputs_actor = self.n_agent_actions[0]
        n_outputs_critic = 1

        self.actor_config["architecture"] = (
            [n_inputs] + self.actor_config["hidden_layers"] + [n_outputs_actor]
        )
        self.critic_config["architecture"] = (
            [n_inputs] + self.critic_config["hidden_layers"] + [n_outputs_critic]
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

        # Critic
        # decentralised: can only observe component state/belief
        # but parameters are shared
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

        belief = self.env.multiagent_idx_percept(belief)
        next_belief = self.env.multiagent_idx_percept(next_belief)

        return super().process_experience(
            belief, action, action_prob, next_belief, reward, terminated, truncated
        )

    def get_future_values(self, t_next_beliefs):

        # get future values
        # shape: (batch_size, num_components)
        future_values = self.critic.forward(t_next_beliefs).squeeze(-1)

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

        return weights.detach().reshape(-1, 1)

    def compute_loss(self, *args):

        # preprocess inputs
        (
            t_ma_beliefs,
            t_ma_next_beliefs,
            t_actions,
            t_action_probs,
            t_rewards,
            t_terminations,
            t_truncations,
        ) = self._preprocess_inputs(*args)

        # Value function update
        current_values = self.critic.forward(
            t_ma_beliefs
        ).squeeze(-1)  # shape: (batch_size, num_components)
        td_targets = self.compute_td_target(
            t_ma_next_beliefs, t_rewards, t_terminations, t_truncations
        )  # shape: (batch_size, num_components)

        advantage = (
            td_targets - current_values
        ).detach()  # shape: (batch_size, num_components)

        # compute log_prob actions
        # shape: (batch_size, n_components)
        t_log_probs = self.compute_log_prob(t_ma_beliefs, t_actions)

        # compute joint probs
        # sum over all actions for each component
        # shape: (batch_size)
        t_joint_log_probs = torch.sum(t_log_probs, dim=-1)

        # shape: (batch_size, 1)
        weights = self.compute_sample_weight(t_joint_log_probs.detach(), t_action_probs)

        # L_V(theta) = E[w (TD_target - V(b))^2]
        # weights: (B, 1), current_values: (B, M), td_targets: (B, M)
        # compute the BMSE across batches
        # weights * (current_values - td_targets)^2
        # (B, 1)  * ((B, M) - (B, M))^2 => (B, M)
        # we take the mean across batches and add losses of all critics
        critic_loss = torch.mean(
            torch.square(current_values - td_targets) * weights, dim=0
        ).sum()

        # t_log_probs @ advantage.T
        # (B, M) * (B, M) => (B, M)
        # torch.sum(B, M, dim=1,keepdim=True) => (B, 1)
        # torch.mean(-(B, 1) * (B, 1)) => (scalar)
        actor_loss = torch.mean(
            -torch.sum(t_log_probs * advantage, dim=1, keepdim=True) * weights
        )

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
