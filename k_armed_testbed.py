import numpy
from numpy.random import normal


class k_armed_testbed():
    INCREMENT_STANDARD_DEVIATION = 0.01

    def __init__(self, k, stationary = True):
        self.k = k
        self.stationary = stationary

        if self.stationary:
            self.action_values = normal(loc = 0, scale = 1, size = self.k)  
        else:
            self.action_values = numpy.full(self.k, fill_value = 0.0)

    def increment_action_values(self):
        if !self.stationary:
            increment = normal(loc = 0, scale = INCREMENT_STANDARD_DEVIATION, size = self.k)
            self.action_values += increment

    def get_action_reward(self, action):
        return normal(loc = self.action_values[action], scale = 1, size = 1)[0]

    def is_optimal_action(self, action):
        return numpy.argmax(self.action_values) == action

    def __str__(self):
        return "\t".join(["Action %d: %.2f" % (action, self.action_values[action]) for action in range(self.k)])