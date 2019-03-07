from abc import ABC, abstractmethod

import numpy as np
import torch
import random
from torch import optim

from model import QNetwork
from memory import ReplayBuffer


class Agent(ABC):
    """General agent that interacts with and learns from the environment."""

    def __init__(self,
                 main_model: QNetwork, target_network: QNetwork,
                 lr=1e-3, batch_size=64,
                 update_every_steps=10,
                 memory_size=int(1e5),
                 seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Initialization
        :param state_size: how many states in world.
        :param action_size: how many agents can the agent choose from.
        :param is_double: if True, implements Double DQN (https://arxiv.org/pdf/1509.06461.pdf)
        :param seed: to reproduce results.
        :param device: do I have a GPU, or not?
        """
        assert main_model.state_size == target_network.state_size, \
            "Main model accepts a state size %d, but target accepts a state size %d" % (main_model.state_size, target_network.state_size)
        assert main_model.action_size == target_network.action_size, \
            "Main model generates %d possible actions, but target generates %d" % (main_model.action_size, target_network.action_size)
        self.version = "v_debug: use of 'target' when calculating targets"
        self.state_size = main_model.state_size
        self.action_size = main_model.action_size
        self.seed = random.seed(seed)
        self.device = device
        self.update_every_steps = update_every_steps
        self.qnetwork_local = main_model.to(self.device)
        self.qnetwork_target = target_network.to(self.device)
        self.optimizer = optim.Adam(params=self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size=memory_size, batch_size=batch_size)
        # let's keep track of the steps so that we can run the algorithms properly
        self.t_step = 0

    def set_models_from(self, torch_file_name):
        """
        Loads weights for networks.
        :param torch_file_name:
        :return:
        """
        self.qnetwork_local.load_state_dict(torch.load(torch_file_name))
        self.qnetwork_target.load_state_dict(torch.load(torch_file_name))

    def step(self, state, action, reward, next_state, done):
        """ One full interaction with the environment.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: True if we ran the learning step; False otherwise.
        """
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.t_step += 1
        if self.t_step % self.update_every_steps == 0:
            if len(self.memory) >= self.memory.batch_size:
                self.learn(self.memory.sample())
                return True
        return False

    @abstractmethod
    def learn(self, experiences, gamma=0.9, tau=1e-3):
        """

        :param experiences:
        :param gamma:
        :return:
        """
        pass

    def act(self, state, eps=0.1):
        """

        :param state: state(s) to be evaluated. Can be as a list, as a numpy array or as tensor.
        :param eps: value in [0,1] for epsilon-greedy choice.
        :return: a numpy array with the actions to take
        """
        # epsilon-greedy choice:
        if random.random() < eps:
            return random.choice(range(self.action_size)) # or np.arange? Does it make a difference?
        # if sate is a list, transform it into a numpy array:
        if isinstance(state, list):
            state = np.asarray(state, dtype=np.float32)
        # if state is a numpy array, transform it into a tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0)
        # we need to work with a Tensor
        if not isinstance(state, torch.FloatTensor):
            raise TypeError('State has to be either a list, a numpy array, or a tensor in this function')
        # let's send this data to the proper device
        state = state.to(self.device)
        # let's put the nn into eval mode:
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # let's get it back as training mode:
        self.qnetwork_local.train()
        # and now let's take the max of this, and return it as a numpy array:
        return np.argmax(action_values.cpu().data.numpy())

class DQNAgent(Agent):
    """General agent that interacts with and learns from the environment."""

    def __init__(self,
                 main_model: QNetwork, target_network: QNetwork,
                 is_double=False,
                 lr=1e-3, batch_size=64,
                 update_every_steps=10,
                 memory_size=int(1e5),
                 seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Agent.__init__(
            self,
            main_model=main_model,
            target_network=target_network,
            lr=lr,
            batch_size=batch_size,
            update_every_steps=update_every_steps,
            memory_size=memory_size,
            seed=seed,
            device=device)

    def learn(self, experiences, gamma=0.9, tau=1e-3):
        """

        :param experiences:
        :param gamma:
        :return:
        """
        # unpack:
        states, actions, rewards, next_states, dones = experiences
        # let's see what is the expected returns of next_states, and take the max of them:
        expected_next = self.qnetwork_target(next_states).detach()  # I don't want the computation graph to keep track of this
        expected_next = expected_next.max(1)[0]  # max values
        # print(expected_next)
        expected_next = expected_next.unsqueeze(1)  # values in columns
        targets = rewards + (gamma * expected_next * (1 - dones))  # in case there is NO 'next'
        # ok, now optimize my network:
        self.qnetwork_local.do_optimization_step(self.optimizer, states_seen=states, actions_taken=actions, targets=targets)

        # let's soft-copy the values onto the target network:
        # target_weights = tau * local_weights = (1 - tau)*target_weights
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DoubleDQNAgent(Agent):
    """General agent that interacts with and learns from the environment."""

    def __init__(self,
                 main_model: QNetwork, target_network: QNetwork,
                 is_double=False,
                 lr=1e-3, batch_size=64,
                 update_every_steps=10,
                 memory_size=int(1e5),
                 seed=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        Agent.__init__(
            self,
            main_model=main_model,
            target_network=target_network,
            lr=lr,
            batch_size=batch_size,
            update_every_steps=update_every_steps,
            memory_size=memory_size,
            seed=seed,
            device=device)

    def learn(self, experiences, gamma=0.9, tau=1e-3):
        """

        :param experiences:
        :param gamma:
        :return:
        """
        # unpack:
        states, actions, rewards, next_states, dones = experiences
        # let's see what is the expected returns of next_states, and take the max of them:
        expected_next_in_local = self.qnetwork_local(next_states).detach()  # I don't want the computation graph to keep track of this
        # print(expected_next_in_   local)
        actions_selected = torch.argmax(expected_next_in_local, dim=1).unsqueeze(1)
        # print(actions_selected)
        expected_next_in_target = self.qnetwork_target(next_states).detach()  # I don't want the computation graph to keep track of this
        # print(expected_next_in_target)
        expected_next = expected_next_in_target.gather(1, actions_selected.long())
        # print("expected_next", expected_next)
        expected_next = torch.transpose(expected_next, 0, 1).data[0]
        # print("expected_next.transposed()", expected_next)
        # print(expected_next)
        expected_next = expected_next.unsqueeze(1)  # values in columns
        targets = rewards + (gamma * expected_next * (1 - dones))  # in case there is NO 'next'
        # ok, now optimize my network:
        self.qnetwork_local.do_optimization_step(self.optimizer, states_seen=states, actions_taken=actions, targets=targets)

        # let's soft-copy the values onto the target network:
        # target_weights = tau * local_weights = (1 - tau)*target_weights
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

