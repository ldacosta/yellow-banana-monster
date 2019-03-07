from abc import ABC, abstractmethod

from collections import deque, namedtuple
import random
import numpy as np
import torch
from numpy.random import choice as np_choice

class RootReplayBuffer(ABC):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Initialize an instance.

        :param buffer_size: maximum size of buffer
        :param batch_size: size of training batch
        :param seed: a random seed
        """
        self.device = device
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)

    def empty(self):
        """ Empties the memory"""
        self.memory.clear()

    @abstractmethod
    def sample(self):
        """Samples a random batch of experiences."""
        pass

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(RootReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Initialize an instance.

        :param buffer_size: maximum size of buffer
        :param batch_size: size of training batch
        :param seed: a random seed
        """
        RootReplayBuffer.__init__(self, buffer_size=buffer_size, batch_size=batch_size, seed=seed, device=device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience into memory."""
        self.memory.append(
            self.experience(state=state, action=int(action), reward=reward, done=int(done), next_state=next_state))

    def sample(self):
        """Samples a random batch of experiences."""
        experiences = random.sample(population=self.memory, k=self.batch_size)

        # let's convert that into tensors, sent into the appropriate device.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class WeightedReplayBuffer(RootReplayBuffer):
    """Replay Buffer where every experience tuple has a weight (in order to do weighted sampling)."""
    def __init__(self, buffer_size, batch_size, seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        RootReplayBuffer.__init__(self, buffer_size=buffer_size, batch_size=batch_size, seed=seed, device=device)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state", "weight"])

    def add(self, state, action, reward, next_state, done, weight):
        """Add a new experience into memory."""
        self.memory.append(
            self.experience(
                state=state,
                action=int(action),
                reward=reward,
                done=int(done),
                next_state=next_state,
                weight=weight))

    def sample(self):
        """Samples a random batch of experiences."""
        all_weights = [e.weight for e in self.memory if e is not None]
        s = sum(all_weights)
        idxs = np_choice(len(self.memory), self.batch_size, p=[w/s for w in all_weights], replace=False)
        experiences = [self.memory[i] for i in idxs]

        # let's convert that into tensors, sent into the appropriate device.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        return (states, actions, rewards, next_states, dones)


