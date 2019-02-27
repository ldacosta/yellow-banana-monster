import unittest
from memory import ReplayBuffer
from random import sample, choice, randint


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.rb = ReplayBuffer(buffer_size=1000, batch_size=64)

    def test_len(self):
        """ Simple test for length."""
        for b in sample(range(100), 3):
            for bs in sample(range(100), 3):
                rb = ReplayBuffer(buffer_size=b, batch_size=bs)
                # len at beginning is 0
                self.assertEqual(len(rb), 0)
                # after adding 1 element, length is 1
                rb.add(state=1, action=1, reward=1, next_state=1, done=1)
                self.assertEqual(len(rb), 1)

    def test_sample(self):
        """ Tests sampling from memory"""
        # let's fill it up first
        for reps in range(10):
            self.rb.empty()
            self.assertEqual(len(self.rb), 0)
            number_of_adds = choice(range(1, 2 * self.rb.batch_size, 1))
            for i in range(number_of_adds):
                self.rb.add(state=randint(1, 100), action=randint(1, 100), reward=randint(0, 1), next_state=randint(1, 100), done=randint(0, 1))
            if number_of_adds < self.rb.batch_size:
                with self.assertRaises(ValueError):
                    (states, actions, rewards, next_states, dones) = self.rb.sample()
            else:
                (states, actions, rewards, next_states, dones) = self.rb.sample()
                self.assertTrue(len(states) == len(actions) == len(rewards) == len(next_states) == len(dones) == self.rb.batch_size)





if __name__ == '__main__':
    unittest.main()
