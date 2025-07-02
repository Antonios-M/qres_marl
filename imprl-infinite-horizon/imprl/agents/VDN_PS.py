import torch

from imprl.agents.primitives.Value_agent import ValueAgent
from imprl.agents.primitives.MLP import NeuralNetwork


class ValueDecompositionNetworkParameterSharing(ValueAgent):
    name = "VDN-PS"
    full_name = "Value-Decomposition Network with Parameter Sharing"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.n_agent_actions = [space.n for space in env.action_space]
        self.n_agents = len(self.n_agent_actions)

        # Neural networks
        # assume homogeneous local observations
        # shape: (local_obs+shared_obs) + n_agents (id)
        obs, info = env.reset()
        ma_idx_obs = env.multiagent_idx_percept(obs)
        n_inputs = ma_idx_obs.shape[1]

        self.network_config["architecture"] = (
            [n_inputs]
            + self.network_config["hidden_layers"]
            + [self.n_agent_actions[0]]
        )

        self.q_network = NeuralNetwork(
            self.network_config["architecture"],
            initialization="orthogonal",
            optimizer=self.network_config["optimizer"],
            learning_rate=self.network_config["lr"],
            lr_scheduler=self.network_config["lr_scheduler"],
        ).to(device)

        self.target_network = NeuralNetwork(self.network_config["architecture"]).to(
            device
        )

        # set weights equal
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialization
        self.target_network_reset = config["TARGET_NETWORK_RESET"]

        self.logger = {
            "TD_loss": None,
            "learning_rate": self.network_config["lr"],
        }

    def get_random_action(self):
        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)

        return action, t_action

    def get_greedy_action(self, observation, training):

        # compute Q-values
        ma_idx_obs = self.env.multiagent_idx_percept(observation)
        t_ma_obs = torch.tensor(ma_idx_obs).to(self.device)

        # shape: (n_agents, n_agent_actions)
        q_values = self.q_network.forward(t_ma_obs, training).squeeze()
        t_action = torch.argmax(q_values, dim=-1)

        action = t_action.cpu().numpy()

        if training:
            return action, t_action
        else:
            return action

    def process_experience(
        self, belief, action, next_belief, reward, terminated, truncated
    ):

        belief = self.env.multiagent_idx_percept(belief)
        next_belief = self.env.multiagent_idx_percept(next_belief)

        return super().process_experience(
            belief, action, next_belief, reward, terminated, truncated
        )

    def mixer(self, q_values):
        # input shape: (batch_size, n_agents)
        return torch.sum(q_values, dim=1)

    def compute_current_values(self, t_ma_obs, t_actions):

        # shape: (batch_size, n_agents, n_agent_actions)
        all_q_values = self.q_network.forward(t_ma_obs)

        q_values = torch.gather(all_q_values, 2, t_actions.unsqueeze(2))

        q_total = self.mixer(q_values)

        return q_total

    def get_future_values(self, t_ma_next_obs):

        # compute Q-values using Q-network
        # shape: (batch_size, n_agents, n_agent_actions)
        q_values = self.target_network.forward(t_ma_next_obs).detach()

        # compute argmax_a Q(s', a)
        # shape: (batch_size, n_agents)
        t_best_actions = torch.argmax(q_values, dim=2)

        # compute Q-values using *target* network
        # shape: (batch_size, n_agents, n_agent_actions)
        target_q_values = self.target_network.forward(t_ma_next_obs).detach()

        # select values correspoding to best actions
        # shape: (batch_size, n_agents)
        future_values = torch.gather(target_q_values, 2, t_best_actions.unsqueeze(2))

        q_total_future = self.mixer(future_values)

        return q_total_future.detach()

    def compute_loss(self, *args):

        # preprocess inputs
        (
            t_ma_obs,
            t_actions,
            t_ma_next_obs,
            t_rewards,
            t_terminations,
            t_truncations,
        ) = self._preprocess_inputs(*args)

        td_target = self.compute_td_target(
            t_ma_next_obs, t_rewards, t_terminations, t_truncations
        )

        current_values = self.compute_current_values(t_ma_obs, t_actions)

        loss = self.q_network.loss_function(current_values, td_target)

        return loss

    def save_weights(self, path, episode):
        torch.save(self.q_network.state_dict(), f"{path}/q_network_{episode}.pth")

    def load_weights(self, path, episode):
        full_path = f"{path}/q_network_{episode}.pth"
        self.q_network.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )