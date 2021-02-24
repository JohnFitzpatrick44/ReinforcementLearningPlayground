import numpy
from tqdm import tqdm
from estimators import SampleAverageEstimator, WeightedEstimator
from k_armed_testbed import KArmedTestbed
from plotter import Plotter

K = 10
STEPS = 1000
RUNS = 2000
INITIAL_ACTION_VALUE_ESTIMATE = 0.0
REWARD_VARIANCE = 1.0
STATIONARY = True

rewards = numpy.full((3, RUNS, STEPS), fill_value = 0.0)
optimal_selections = numpy.full((3, RUNS, STEPS), fill_value = 0.0)

for run in tqdm(range(RUNS)):

    testbed = KArmedTestbed(k = K, reward_variance = REWARD_VARIANCE, stationary = STATIONARY)

    action_value_estimates = numpy.full(K, fill_value = INITIAL_ACTION_VALUE_ESTIMATE)

    greedy_estimator = SampleAverageEstimator(action_value_estimates.copy(), epsilon = 0.0)
    sample_average_estimator_1 = SampleAverageEstimator(action_value_estimates.copy(), epsilon = 0.01)
    sample_average_estimator_2 = SampleAverageEstimator(action_value_estimates.copy(), epsilon = 0.1)

    estimators = [greedy_estimator, sample_average_estimator_1, sample_average_estimator_2]

    for step in range(STEPS):
        for estimator_index, estimator in enumerate(estimators):
            action = estimator.select_action()
            is_optimal = testbed.is_optimal_action(action)
            reward = testbed.get_action_reward(action)
            estimator.update_estimates(action, reward)

            rewards[estimator_index][run][step] = reward
            optimal_selections[estimator_index][run][step] = is_optimal

        testbed.update_action_values()

Plotter.make_average_run_reward_plot(["Ɛ=0 (greedy)", "Ɛ=0.01", "Ɛ=0.1"], numpy.array(rewards))
Plotter.make_optimal_selection_plot(["Ɛ=0 (greedy)", "Ɛ=0.01", "Ɛ=0.1"], numpy.array(optimal_selections))