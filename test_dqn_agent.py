import unittest
from dqn_agent import Agent
from random import sample, choice, randint, random
import numpy as np


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(state_size=5, action_size=3)
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
