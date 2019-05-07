# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey


class Learner(object):

    def __init__(self):
        self.last_state  = 0
        self.last_action = 0
        self.last_reward = 0
        self.gamma = 0.8
        self.alpha = 0.2
        self.Q = [[0,0] for i in range(10000)] 
        self.epsilon = 0.1

    def reset(self):
        self.last_state  = 0
        self.last_action = 0
        self.last_reward = 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.
        
        # Discretizing into 60x40 rectangles
        
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        
        vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        
        tree_top_discretized = tree_top//40
        tree_bot_discretized = tree_bot//40
        
        monkey_top_discretized = monkey_top//40
        monkey_bot_discretized = monkey_bot//40
        
        vel_discretized = vel//40
        
        tree_dist_discretized = tree_dist//60
        
        top_dist = tree_top_discretized - monkey_top_discretized
        bot_dist = tree_bot_discretized - monkey_bot_discretized
        
        state_vec = [tree_dist_discretized, top_dist, bot_dist, vel_discretized]
        state_rep = int(state_vec[0] * (10**3) + state_vec[1] * (10**2) 
                    + state_vec[2] * (10) + state_vec[3])
        
        #Choosing with epsilon greedy
        a = -1.3
        U = np.random.uniform()
        if U <= 1 - self.epsilon:
            a = np.argmax(np.array(self.Q[self.last_state]))
        else:
            a = np.random.choice([0,1])
        
        self.Q[self.last_state][a] = self.Q[self.last_state][a] + self.alpha*(self.last_reward +
                     self.gamma * float(max(self.Q[state_rep])) - float(self.Q[self.last_state][a]))
        
        self.last_state = state_rep

        self.last_action = np.argmax(np.array(self.Q[state_rep]))
        
        # Adaptive epsilon greedy
        self.epsilon -= 0.0005*self.epsilon

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
    agent = Learner()

	# Empty list to save history.
    hist = []

	# Run games. 
    run_games(agent, hist, 500, 1)

	# Save history. 
    np.save('hist',np.array(hist))
    print(max(hist))
    plt.plot(hist)
    plt.show()

