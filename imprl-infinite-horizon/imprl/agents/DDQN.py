import torch

from imprl.agents.primitives.Value_agent import ValueAgent
from imprl.agents.primitives.MLP import NeuralNetwork


class DDQNAgent(ValueAgent):
    name = "DDQN"
    full_name = "Double Deep Q-Network"

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        assert env.single_agent == False, "DDQN only supports multi-agent environments"
        ## Neural Networks
        n_inputs = self.env.perception_dim
        n_outputs = self.env.action_dim

        self.network_config["architecture"] = (
            [n_inputs] + self.network_config["hidden_layers"] + [n_outputs]
        )

        # initialise Q network and target network
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

        return action, torch.tensor(action).to(self.device)

    def get_greedy_action(self, observation, training):

        # compute Q values
        flat_obs = self.env.system_percept(observation)
        t_observation = torch.tensor(flat_obs).to(self.device)
        q_values = self.q_network.forward(t_observation, training)

        t_action = torch.argmax(q_values)

        action = t_action.cpu().detach().numpy()

        if training:
            return action, t_action
        else:
            return action

    def process_experience(self, belief, action, next_belief, reward, terminated, truncated):

        belief = self.env.system_percept(belief)
        next_belief = self.env.system_percept(next_belief)

        super().process_experience(belief, action, next_belief, reward, terminated, truncated)

    def compute_current_values(self, t_observations, t_actions):

        q_values = self.q_network.forward(t_observations)

        return torch.gather(q_values, dim=1, index=t_actions.unsqueeze_(1))

    def get_future_values(self, t_next_beliefs):

        # compute Q-values using Q-network
        q_values = self.q_network.forward(t_next_beliefs).detach()

        # compute argmax over actions
        t_idx_best_actions = torch.argmax(q_values, axis=1, keepdim=True)

        # compute future Q-values using *target* network
        target_q_values = self.target_network.forward(t_next_beliefs).detach()

        # select values correspoding to best actions
        future_values = torch.gather(target_q_values, dim=1, index=t_idx_best_actions)

        return future_values.detach()

    def compute_loss(self, *args):

        # preprocess inputs
        (
            t_beliefs,
            t_actions,
            t_next_beliefs,
            t_rewards,
            t_terminations,
            t_truncations,
        ) = self._preprocess_inputs(*args)

        td_targets = self.compute_td_target(
            t_next_beliefs, t_rewards, t_terminations, t_truncations
        )

        current_values = self.compute_current_values(t_beliefs, t_actions)

        loss = self.q_network.loss_function(current_values, td_targets)

        return loss

    def save_weights(self, path, episode):
        torch.save(self.q_network.state_dict(), f"{path}/q_network_{episode}.pth")

    def load_weights(self, path, episode):
        full_path = f"{path}/q_network_{episode}.pth"
        self.q_network.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )
