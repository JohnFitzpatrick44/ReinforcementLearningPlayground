import matplotlib.pyplot as plot
import numpy

class plotter():
    def make_average_run_reward_plot(run_names, rewards):
        for i, plot_names in enumerate(plot_names):
            average_run_rewards = numpy.average(rewards[i], axis = 0)
            plot.plot(average_run_rewards, label = plot_names)

        plot.legend()
        plot.xlabel("Time Steps")
        plot.ylabel("Average Reward")
        plot.show()

    def make_optimal_selection_plot(plot_names, optimal_selections):
        for i, plot_names in enumerate(plot_names):
            average_run_optimal_percentage = numpy.average(optimal_selections[i], axis = 0)
            plot.plot(average_run_optimal_percentage, label = plot_names)
        plot.legend()
        plot.xlabel("Time Steps")
        plot.ylabel("Optimal Action Percentage")
        plot.show()

    def make_parameter_study_plot(parameter_values, plot_names, rewards):
        for estimator, estimator_results in enumerate(rewards):
            average_parameter_results = []
            for parameter_setting_results in estimator_results:
                average_run_results = numpy.average(parameter_setting_results, axis = 0)
                average_parameter_results.append(numpy.average(average_run_results))

            plot.plot(parameter_values, average_parameter_results, label = plot_names[estimator])

        plot.legend()
        plot.xlabel("ε, α, c, Q0")
        plot.xscale("log", basex = 2)
        plot.ylabel("Average reward over last 100000 steps")
        plot.show()

