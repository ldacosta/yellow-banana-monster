import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class QNetwork(nn.Module):
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
        # self.hidden_sizes = [int(round(state_size * .8)), int(round(state_size * .6)), int(round(action_size * 2))]
        # self.hidden_sizes = [int(round(state_size * 10)), int(round(state_size * 5)), int(round(action_size * 10))]
        self.hidden_sizes = [int(round(state_size * 1.5)), int(round(action_size * 1.5))]
        print("Creating network '%s' with %d input neurons, %d output, and hidden of %d and %d" %
              (self.name, self.state_size, self.action_size, self.hidden_sizes[0], self.hidden_sizes[1]))
        self.fc1 = nn.Linear(self.state_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        # self.fc3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.output = nn.Linear(self.hidden_sizes[1], self.action_size)

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
        first_layer_result = F.relu(self.fc1(state))
        second_layer_result = F.relu(self.fc2(first_layer_result))
        # third_layer_result = F.relu(self.fc3(second_layer_result))
        return self.output(second_layer_result)

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
