import numpy as np
import torch

from imprl.agents.primitives.agent import Agent


class ValueAgent(Agent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.network_config = config["NETWORK_CONFIG"]

        self.logger = {
            "TD_loss": None,
            "learning_rate": self.network_config["lr"],
        }

    def reset_episode(self, training=True):

        super().reset_episode(training)

        # if training and sufficient samples are available
        if training and self.total_time > 10 * self.batch_size:

            # set weights equal
            if self.episode % self.target_network_reset == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # update learning rate
            self.q_network.lr_scheduler.step()

            # logging
            self.logger["learning_rate"] = self.q_network.lr_scheduler.get_last_lr()[0]

    def process_experience(
        self, belief, action, next_belief, reward, terminated, truncated
    ):

        super().process_rewards(reward)

        # store experience in replay memory
        self.replay_memory.store_experience(
            belief, action, next_belief, reward, terminated, truncated
        )

        # start batch learning once sufficient samples are available
        if self.total_time > 10 * self.batch_size:
            sample_batch = self.replay_memory.sample_batch(self.batch_size)
            self.train(*sample_batch)

        if terminated or truncated:
            self.logger["episode"] = self.episode
            self.logger["episode_returns"] = self.episode_return

    def train(self, *args):

        loss = self.compute_loss(*args)

        # Zero gradients, perform a backward pass, and update the weights.
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        # logging value update
        self.logger["TD_loss"] = loss.detach()

    def _preprocess_inputs(
        self, beliefs, actions, next_beliefs, rewards, terminations, truncations
    ):

        t_beliefs = torch.tensor(np.array(beliefs)).to(self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_next_beliefs = torch.tensor(np.array(next_beliefs)).to(self.device)
        t_terminations = (
            torch.tensor(terminations, dtype=torch.int).reshape(-1, 1).to(self.device)
        )
        t_truncations = (
            torch.tensor(truncations, dtype=torch.int).reshape(-1, 1).to(self.device)
        )

        return (
            t_beliefs,
            t_actions,
            t_next_beliefs,
            t_rewards,
            t_terminations,
            t_truncations,
        )