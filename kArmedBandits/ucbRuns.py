import numpy
from tqdm import tqdm
from estimators import UCBEstimator
from k_armed_testbed import KArmedTestbed
from plotter import Plotter

K = 10
STEPS = 1000
RUNS = 2000
INITIAL_ACTION_VALUE_ESTIMATE = 0.0
REWARD_VARIANCE = 1.0
STATIONARY = False

rewards = numpy.full((4, RUNS, STEPS), fill_value = 0.0)
optimal_selections = numpy.full((4, RUNS, STEPS), fill_value = 0.0)

for run in tqdm(range(RUNS)):

    testbed = KArmedTestbed(k = K, reward_variance = REWARD_VARIANCE, stationary = STATIONARY)

    action_value_estimates = numpy.full(K, fill_value = INITIAL_ACTION_VALUE_ESTIMATE)

    estimator1 = UCBEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = 0.1, c = 1.5)
    estimator2 = UCBEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = 0.1, c = 1.5)
    estimator3 = UCBEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = 0.5, c = 2.5)
    estimator4 = UCBEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = 0.5, c = 2.5)

    estimators = [estimator1, estimator2, estimator3, estimator4]

    for step in range(STEPS):
        for estimator_index, estimator in enumerate(estimators):
            action = estimator.select_action()
            is_optimal = testbed.is_optimal_action(action)
            reward = testbed.get_action_reward(action)
            estimator.update_estimates(action, reward)

            rewards[estimator_index][run][step] = reward
            optimal_selections[estimator_index][run][step] = is_optimal

        testbed.update_action_values()

Plotter.make_average_run_reward_plot(["Ɛ=0.1, α=0.1, c=1.5", "Ɛ=0.1, α=0.1, c=1.5", "Ɛ=0.1, α=0.5, c=2.5", "Ɛ=0.1, α=0.5, c=2.5"], numpy.array(rewards))
Plotter.make_optimal_selection_plot(["Ɛ=0.1, α=0.1, c=1.5", "Ɛ=0.1, α=0.1, c=1.5", "Ɛ=0.1, α=0.5, c=2.5", "Ɛ=0.1, α=0.5, c=2.5"], numpy.array(optimal_selections))