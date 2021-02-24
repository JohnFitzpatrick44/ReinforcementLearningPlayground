import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
from k_armed_testbed import KArmedTestbed
from estimators import SampleAverageEstimator, WeightedEstimator, GradientEstimator, UCBEstimator
from plotter import Plotter

K = 10
STEPS = 100000
RUNS = 5

starting_index = STEPS/2

parameter_values = [1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0]

rewards = numpy.full((4, len(parameter_values), RUNS, STEPS/2), fill_value = 0.0)

for parameter_index, parameter_value in enumerate(parameter_values):
    for run in range(RUNS):
        testbed = KArmedTestbed(k = K, stationary = False)

        action_value_estimates = numpy.full(K, fill_value = 0.0)
        sample_average_estimator = SampleAverageEstimator(action_value_estimates.copy(), epsilon = parameter_value)
        weighted_estimator = WeightedEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = parameter_value)
        ucb_estimator = UCBEstimator(action_value_estimates.copy(), epsilon = 0.1, alpha = 0.1, c = parameter_value)
        gradient_estimator = GradientEstimator(action_value_estimates.copy(), alpha = parameter_value)

        estimators = [sample_average_estimator, weighted_estimator, ucb_estimator, gradient_estimator]

        for step in tqdm(range(STEPS)):
            for estimator_index, estimator in enumerate(estimators):
                action_selected = estimator.select_action()
                reward = testbed.get_action_reward(action_selected)
                estimator.update_estimates(action_selected, reward)

                if step >= starting_index:
                    rewards[estimator_index][parameter_index][run][step - starting_index] = reward

            testbed.update_action_values()

estimator_names = ["Sample Average Estimator", "Constant Step-size Estimator", "UCB Estimator", "Gradient Estimator"]
Plotter.make_parameter_study_plot(parameter_values, estimator_names, rewards)