# yellow-banana-monster
Chase the yellow bananas on an Unity environment


## Description

Training of an agent to navigate (and collect bananas!) in a large, square world. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

1. move forward.
2. move backward.
3. turn left.
4. turn right.

The task is episodic, and in order to solve the environment an agent must get an average score of +13 over 100 consecutive episodes.

### Note

The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6. (replace `my_env` below with a name of your choosing).

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate my_env
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate my_env
	```

## How to Run it

If you'd like to explore this solution open jupyter's notebook
```jupyter notebook```
from the root of this repository, open the notebook called `Navigation.ipynb` and follow ths instructions on it.