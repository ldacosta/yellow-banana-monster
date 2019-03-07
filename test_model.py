import unittest
from model import QNetwork
import numpy as np
import random
from torch import nn


class TestModel(unittest.TestCase):

    def setUp(self):
        self.state_size = 3
        self.action_size = 5
        fc = nn.Sequential(
            nn.Linear(self.state_size, 5),
            nn.ReLU(),
            nn.Linear(5, 7),
            nn.ReLU(),
            nn.Linear(7, 9),
            nn.ReLU(),
            nn.Linear(9, self.action_size)
        )
        self.model = QNetwork(name="my_network", fc=fc)

    def test_forward(self):
        """Minimally."""
        # let's see if it runs till the end
        state_value = [random.random()] * self.model.state_size
        result = self.model.forward_np(numpy_state=np.asarray(state_value, dtype=np.float32))
        # ok. If I call it with the wrong number of arguments, I want a quick error:
        state_value = [1] * (self.model.state_size + 1)  # list too long for the number of states
        with self.assertRaises(ValueError):
            result = self.model.forward_np(numpy_state=np.asarray(state_value, dtype=np.float32))

