import unittest
from dqn_agent import Agent
from random import random
import numpy as np
from torch import nn
from model import QNetwork


class TestAgent(unittest.TestCase):

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
        self.main_model = QNetwork(name="my_network", fc=fc)
        self.target_model = QNetwork(name="my_network", fc=fc)
        self.agent = Agent(main_model=self.main_model, target_network=self.target_model)
        self.eps_greediness = 0.01

    def test_allruns(self):
        """ No explosions? """
        # act
        state_value = [random()] * self.agent.state_size
        self.agent.act(state=state_value, eps=self.eps_greediness)

        agent_learned = False
        while not agent_learned:  # I want to force a learning step.
            agent_learned = self.agent.step(
                state=[random()] * self.agent.state_size,
                action=np.random.randint(self.agent.action_size) ,
                reward=random(),
                next_state=[random()] * self.agent.state_size,
                done=random() > 0.75
            )


if __name__ == '__main__':
    unittest.main()
