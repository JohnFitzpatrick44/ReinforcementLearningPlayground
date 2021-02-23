import numpy


class estimator():
    def __init__(self, action_value_initial_estimates):
        self.action_value_estimates = action_value_initial_estimates
        self.k = len(action_value_initial_estimates)
        self.action_selection_count = numpy.full(self.k, fill_value = 0)

    def select_action(self):
        raise NotImplementedError("Action selection not implemented.")

    def update_estimates(self):
        raise NotImplementedError("Estimate update not implemented.")

    def select_greedy_action(self):
        return numpy.argmax(self.action_value_estimates)

    def select_exploration_action(self):
        return np.random.choice(self.k)


class sample_average_estimator(estimator):
    def __init__(self, action_value_initial_estimates, epsilon):
        super(self).__init__(action_value_initial_estimates)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() >= self.epsilon:
            return self.select_greedy_action()

        return self.select_exploration_action()

    def update_estimates(self, action, reward):
        self.action_selection_count[action] += 1

        qn = self.action_value_estimates[action]
        n = self.action_selection_count[action]

        self.action_value_estimates[action] = qn + (1.0 / n) * (reward - qn)
        
