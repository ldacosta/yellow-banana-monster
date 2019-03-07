import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor Model: consitutes a Policy, as it gets state and returns the optimal action."""

    def __init__(self, name, fc: nn.Sequential, seed=0):
        """
        Defines model.
        :param name:
        :param fc:
        :param state_size: number of states
        :param action_size: number of actions
        :param seed: seed for random initialization

        fc can be, eg,
        nn.Sequential(
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            )
        """
        super().__init__()
        self.name = name
        self.seed = torch.manual_seed(seed)
        self.fc = fc
        self.state_size = self.fc[0].in_features
        self.action_size = self.fc[-1].out_features

    def forward_np(self, numpy_state):
        """

        So if you had a numpy vector that you'd like to pass to this function
        v = [ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]
        you'd need to:
        1. convert it to a tensor: v = torch.from_numpy(v)
        2. just for good measure, get it out of numpy.float64 or other native type: v = v.float()
        3. at this point you have => v = tensor([ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000])
        4. so: v = v.unsqueeze(0) => v = tensor([[ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]])

        :param state_as_array:
        :return:
        """
        if isinstance(numpy_state, list):
            numpy_state = np.asarray(numpy_state)
        if not isinstance(numpy_state, np.ndarray):
            raise TypeError('State has to be a numpy array in this function')
        if len(numpy_state) != self.state_size:
            raise ValueError(
                'This network is expecting an input state size %d, but this function is called with structure size %d'
                % (self.state_size, len(numpy_state)))
        state = torch.from_numpy(numpy_state)  # state is a tensor
        state = state.float()  # numpy has a type np.float64. I want a direct float
        state = state.unsqueeze(0)
        return self.forward(state)

    def forward(self, state):
        """
        Makes the map state(s) -> action(s)
        :param state: a 2D tensor. So a stack of states, something like
        tensor([[ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]])
        :return: a tensor with the proposed actions.
        :seealso: if you have a list of states as numpy array, see 'forward_np'
        """
        return self.fc(state)

    def do_optimization_step(self, optimizer, states_seen, actions_taken, targets):
        """

        :param states_seen:
        :param actions_taken:
        :param targets:
        :return:
        """
        # compute actual rewards obtained
        rewards_obtained = self.forward(states_seen).gather(1, actions_taken.long())
        loss = F.mse_loss(input=rewards_obtained, target=targets) # compute the loss
        # take an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class QNetworkOld(nn.Module):
    """Actor Model: consitutes a Policy, as it gets state and returns the optimal action."""

    def __init__(self, name, state_size, action_size, seed=0):
        """
        Defines model.
        :param state_size: number of states
        :param action_size: number of actions
        :param seed: seed for random initialization
        """
        super().__init__()
        self.name = name
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size






        self.tmp_three_layers = True
        if self.tmp_three_layers:
            # 3 hidden layers
            self.hidden_sizes = [int(round(state_size * .8)), int(round(state_size * .6)), int(round(action_size * 2))]
            # self.hidden_sizes = [int(round(state_size * 10)), int(round(state_size * 5)), int(round(action_size * 10))]
            self.fc = nn.Sequential(
                nn.Linear(self.state_size, self.hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[2], self.action_size)
            )
        else:
            # 2 hidden layers
            self.hidden_sizes = [int(round(state_size * 1.5)), int(round(action_size * 1.5))]
            self.fc = nn.Sequential(
                nn.Linear(self.state_size, self.hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[1], self.action_size)
            )
        print("hello")

    def forward_np(self, numpy_state):
        """

        So if you had a numpy vector that you'd like to pass to this function
        v = [ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]
        you'd need to:
        1. convert it to a tensor: v = torch.from_numpy(v)
        2. just for good measure, get it out of numpy.float64 or other native type: v = v.float()
        3. at this point you have => v = tensor([ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000])
        4. so: v = v.unsqueeze(0) => v = tensor([[ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]])

        :param state_as_array:
        :return:
        """
        if not isinstance(numpy_state, np.ndarray):
            raise TypeError('State has to be a numpy array in this function')
        if len(numpy_state) != self.state_size:
            raise ValueError(
                'This network is expecting an input state size %d, but this function is called with structure size %d'
                % (self.state_size, len(numpy_state)))
        state = torch.from_numpy(numpy_state)  # state is a tensor
        state = state.float()  # numpy has a type np.float64. I want a direct float
        state = state.unsqueeze(0)
        return self.forward(state)

    def forward(self, state):
        """
        Makes the map state(s) -> action(s)
        :param state: a 2D tensor. So a stack of states, something like
        tensor([[ 0.0001,  0.9364,  0.0120, -0.2838, -0.0001, -0.0027,  0.0000, 0.0000]])
        :return: a tensor with the proposed actions.
        :seealso: if you have a list of states as numpy array, see 'forward_np'
        """
        return self.fc(state)

    def do_optimization_step(self, optimizer, states_seen, actions_taken, targets):
        """

        :param states_seen:
        :param actions_taken:
        :param targets:
        :return:
        """
        # compute actual rewards obtained
        rewards_obtained = self.forward(states_seen).gather(1, actions_taken.long())
        loss = F.mse_loss(input=rewards_obtained, target=targets) # compute the loss
        # take an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
