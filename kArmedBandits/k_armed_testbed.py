import numpy
from numpy.random import normal


class KArmedTestbed():
    def __init__(self, k, reward_variance = 1.0, stationary = True):
        self.k = k
        self.reward_variance = reward_variance
        self.stationary = stationary

        if self.stationary:
            # Mean action reward value distribution
            self.action_values = normal(loc = 0, scale = 1, size = self.k)  
        else:
            # Mean action reward values start at 0, and randomly increment
            self.action_values = numpy.full(self.k, fill_value = 0.0)

    def update_action_values(self):
        if not self.stationary:
            increment = normal(loc = 0, scale = 0.01, size = self.k)
            self.action_values += increment

    def get_action_reward(self, action):
        return normal(loc = self.action_values[action], scale = self.reward_variance, size = 1)[0]

    def is_optimal_action(self, action):
        return numpy.argmax(self.action_values) == action
