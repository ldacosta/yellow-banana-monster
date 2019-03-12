# Yellow Banana Monster

For this assignment we want to solve the problem of collecting yellow bananas (and avoiding blue ones!) in a Unity-created world (a better description is found on the [README file](README.md)).

This address of this repository is [github.com/ldacosta/yellow-banana-monster](https://github.com/ldacosta/yellow-banana-monster); the branch where this solution is kept is called [drlnd-project1](https://github.com/ldacosta/yellow-banana-monster/tree/drlnd-project1); the most up-to-date implementation is kept on [master](https://github.com/ldacosta/yellow-banana-monster/tree/master), where this example is kept along others.

## Learning Algorithm

Three flavors of Deep Q-Learning were implemented for this work.
 
1. **"Traditional" Deep Q-Network** (DQN) where a target network is maintained on top of the "main" network, and the training is done on a set of saved actions (and their results). The "target" network is kept to give the main network a goal to train for.
2. **Double DQN**, where the "target" network also serves to evaluate actions before taken them.
3. **Prioritized Replay**, where the memory used for training is queried based on the perceived importance of each tuple.

For all these flavors the architecture of the underlying neural network was kept unchanged:

* Vanilla feed-forward, fully-connected network, trained with back-propagation; batch size of 64 samples.
* Target network is soft-updated every 10 steps with $alpha=0.1$
* 37 input neurons(size of a state)
* 3 hidden layers, of size 30, 22, and 12, respectively.
* 4 output neurons (of actions)

 
## Plot of Rewards

Typical run for successfull experience:
![alt text][logo]

[logo]: dqn_solved.png "Successful Run"

As you can see, the mean reward of 13 us about at t=800

## Ideas for Future Work
I am slightly disappointed by the fact that I couldn't make *Double DQN* or *Prioritized Replay* work.

* **Double DQN did not really work**. The number of iterations needed for convergence was higher than for DQN, which contradicts the litterature. Maybe a bug?
* **Prioritized Replay is very slow**. I suspect this has to do something with the fact of shipping data to/from the CPU.