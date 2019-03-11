
import time
import torch
import numpy as np
from collections import deque
from tqdm import tqdm


def run(
        agent,
        env, brain_name,
        n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        feedback_every_secs=30, keep_last_scores=100,
        file_name_where_to_save='checkpoint.pth'):
    """Deep Q-Learning.

    Params
    ======
        agent (Agent): the agent that takes decisions
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        feedback_every_secs (int): how often (in secs.) do you want to have feedback about the scores.
        keep_last_scores (int): I will keep this many scores and then report on their statistics.
    """
    all_scores = []  # ALL scores
    scores_window = deque(maxlen=keep_last_scores)  # last <keep_last_scores> scores
    eps = eps_start                    # initialize epsilon
    last_time = time.time()
    for i_episode in tqdm(range(1, n_episodes +1)):
        # print("Resetting the episode...\n")
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        # print("... environment reset!\n")
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state=state, eps=eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            agent.step(state=state, action=action, reward=reward, next_state=next_state, done=done)
            state = next_state                             # roll over the state to next time step
            if done:
                # print("===> Environment finished this episode after %d steps (max = %d) (ie, 'done' == True)" % (t, max_t))
                break
        scores_window.append(score)       # save most recent score
        all_scores.append(score)
        eps = max(eps_end, eps_decay *eps) # decrease epsilon
        if time.time() - last_time >= feedback_every_secs:
            print('\rEpisode {}, eps: {:.3f}\tAverage Score (last {} episodes): {:.2f}'.format(i_episode, eps, keep_last_scores, np.mean(scores_window)))
            last_time = time.time()
        if np.mean(scores_window )>= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode -100, np.mean(scores_window)))
            break
    print('\rEpisode {}, eps: {:.3f}\tAverage Score (last {} episodes): {:.2f}'.format(i_episode, eps, keep_last_scores, np.mean(scores_window)))
    print("\n Saving best network on '%s'" % (file_name_where_to_save))
    torch.save(agent.qnetwork_local.state_dict(), file_name_where_to_save)
    return all_scores

