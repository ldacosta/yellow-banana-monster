import unittest
from model import QNetwork
import numpy as np
import random


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = QNetwork(name="my_network", state_size=3, action_size=5)

    def test_forward(self):
        """Minimally."""
        # let's see if it runs till the end
        state_value = [random.random()] * self.model.state_size
        result = self.model.forward_np(numpy_state=np.asarray(state_value, dtype=np.float32))
        # ok. If I call it with the wrong number of arguments, I want a quick error:
        state_value = [1] * (self.model.state_size + 1)  # list too long for the number of states
        with self.assertRaises(ValueError):
            result = self.model.forward_np(numpy_state=np.asarray(state_value, dtype=np.float32))

