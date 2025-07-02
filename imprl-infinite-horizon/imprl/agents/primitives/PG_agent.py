import numpy as np
import torch

from imprl.agents.primitives.agent import Agent


class PolicyGradientAgent(Agent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.actor_config = config["ACTOR_CONFIG"]
        self.critic_config = config["CRITIC_CONFIG"]

        # logger
        self.logger = {
            "critic_loss": None,
            "actor_loss": None,
            "lr_critic": self.critic_config["lr"],
            "lr_actor": self.actor_config["lr"],
            "pg_mean": None,
            "pg_variance": None,
        }

    def get_random_action(self):

        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)
        action_prob = torch.prod(1 / torch.tensor(self.n_agent_actions))

        return action, t_action, action_prob

    def reset_episode(self, training=True):

        super().reset_episode(training)

        # if training and sufficient samples are available
        if training and self.total_time > 10 * self.batch_size:

            # update learning rate
            self.actor.lr_scheduler.step()
            self.critic.lr_scheduler.step()

            # logging
            self.logger["lr_actor"] = self.actor.lr_scheduler.get_last_lr()[0]
            self.logger["lr_critic"] = self.critic.lr_scheduler.get_last_lr()[0]

            # compute policy gradient metrics
            if self.episode % 500 == 0:
                for i in [1, 2, 4]:
                    batch_size = i * self.batch_size
                    (
                        self.logger[f"pg_mean_{batch_size}"],
                        self.logger[f"pg_variance_{batch_size}"],
                    ) = self.compute_pg_metrics(batch_size)

    def process_experience(
        self, belief, action, action_prob, next_belief, reward, terminated, truncated
    ):

        super().process_rewards(reward)

        # store experience in replay memory
        self.replay_memory.store_experience(
            belief, action, action_prob, next_belief, reward, terminated, truncated
        )

        # start batch learning once sufficient samples are available
        if self.total_time > 10 * self.batch_size:

            # sample batch of experiences from replay memory
            sample_batch = self.replay_memory.sample_batch(self.batch_size)

            # train actor and critic networks
            self.train(*sample_batch)

        if terminated or truncated:
            self.logger["episode"] = self.episode
            self.logger["episode_return"] = self.episode_return

    def train(self, *args):

        actor_loss, critic_loss = self.compute_loss(*args)

        ## Actor network
        # Zero gradients, perform a backward pass, and update the weights.
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        ## Critic network
        # Zero gradients, perform a backward pass, and update the weights.
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # logging value update
        self.logger["actor_loss"] = actor_loss.detach()
        self.logger["critic_loss"] = critic_loss.detach()

    def _preprocess_inputs(
        self,
        beliefs,
        actions,
        action_probs,
        next_beliefs,
        rewards,
        terminations,
        truncations,
    ):

        t_beliefs = torch.tensor(np.array(beliefs)).to(self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)
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
            t_next_beliefs,
            t_actions,
            t_action_probs,
            t_rewards,
            t_terminations,
            t_truncations,
        )

    def compute_pg_metrics(self, batch_size):

        # Collect gradients
        gradients = []
        for _ in range(batch_size):

            sample_batch = self.replay_memory.sample_batch(1)
            actor_loss, _ = self.compute_loss(*sample_batch)
            actor_loss.backward()

            grads = []

        ## [ISSUE]
        ## grad collection doesnt work for DDMAC:
        ## current implimentation:
        # for param in self.actor.parameters():
        #         if param.grad is not None:
        #             grads.append(param.grad.view(-1))
        ## [FIX]
        if hasattr(self.actor, 'networks'):
            # DDMAC
            for actor in self.actor.networks:
                # Flatten gradients for each actor
                for param in actor.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
        else:
            # DCMAC
            for param in self.actor.parameters():
                if param.grad is not None:
                        grads.append(param.grad.view(-1))
        ## [ISSUE]

        # conc actor grads
        gradients.append(torch.cat(grads))

        # Compute mean and variance
        gradients = torch.stack(gradients)

        return torch.mean(gradients), torch.var(gradients)
