import numpy


class Estimator(object):
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
        return numpy.random.choice(self.k)


class SampleAverageEstimator(Estimator):
    def __init__(self, action_value_initial_estimates, epsilon = 0):
        super(SampleAverageEstimator, self).__init__(action_value_initial_estimates)
        self.epsilon = epsilon

    def select_action(self):
        if numpy.random.rand() >= self.epsilon:
            return self.select_greedy_action()

        return self.select_exploration_action()

    def update_estimates(self, action, reward):
        self.action_selection_count[action] += 1

        qn = self.action_value_estimates[action]
        n = self.action_selection_count[action]

        self.action_value_estimates[action] = qn + (1.0 / n) * (reward - qn)


class WeightedEstimator(SampleAverageEstimator):
    def __init__(self, action_value_initial_estimates, epsilon = 0, alpha = 0.5):
        super(WeightedEstimator, self).__init__(action_value_initial_estimates, epsilon)
        self.alpha = alpha

    def update_estimates(self, action, reward):
        qn = self.action_value_estimates[action]
        self.action_value_estimates[action] = qn + self.alpha * (reward - qn)


class UCBEstimator(WeightedEstimator):
    def __init__(self, action_value_initial_estimates, epsilon = 0, alpha = 0.5, c = 2):
        super(UCBEstimator, self).__init__(action_value_initial_estimates, epsilon, alpha)
        self.c = c
        self.t = 0

    def select_action(self):
        self.t += 1
        if numpy.random.rand() >= self.epsilon:
            return self.select_greedy_action()

        return self.select_ucb_action()

    def get_action_potential(self, action):
        qt = self.action_value_estimates[action]
        ln_t = numpy.log(self.t)
        nt = self.action_selection_count[action]

        return qt + self.c * numpy.sqrt(ln_t / nt)

    def select_ucb_action(self):
        actions_never_selected = [action for action in range(self.k) if self.action_selection_count[action] == 0]
        if len(actions_never_selected) > 0:
            selected_action = numpy.random.choice(actions_never_selected)
            self.action_selection_count[selected_action] += 1
            return selected_action

        action_potential = [self.get_action_potential(action) for action in range(self.k)]
        action_potential[self.select_greedy_action()] = -1

        return numpy.argmax(action_potential)


class GradientEstimator(Estimator):
    def __init__(self, action_value_initial_estimates, alpha):
        super(GradientEstimator, self).__init__(action_value_initial_estimates)
        self.average_reward = 0
        self.numerical_preference = numpy.full(self.k, fill_value = 0.0)
        self.alpha = alpha

    def get_action_probabilities(self):
        exp_ht = numpy.exp(self.numerical_preference)
        return exp_ht / numpy.sum(exp_ht)

    def select_action(self):
        return numpy.random.choice(a = self.k, p = self.get_action_probabilities())

    def update_average_reward(self, reward):
        self.average_reward += self.alpha * (reward - self.average_reward)

    def update_estimates(self, action, reward):
        self.update_average_reward(reward)

        p = self.get_action_probabilities()

        h_next = self.numerical_preference - self.alpha * (reward - self.average_reward) * p
        h_next[action] = self.numerical_preference[action] + self.alpha * (reward - self.average_reward) * (1 - p[action])

        self.numerical_preference = h_next
