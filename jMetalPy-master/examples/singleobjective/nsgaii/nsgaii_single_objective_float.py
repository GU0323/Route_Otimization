from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Linear4
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot

if __name__ == '__main__':
    problem = Linear4()

    max_evaluations = 100000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
        crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator()
    )

    algorithm.run()
    front = algorithm.get_result()
    plot_front = Plot(title = 'Pareto Front',
                      axis_labels = ["x", "y"])
    plot_front.plot(front)

    # Save results to file
    print_function_values_to_file(front, 'FUN.'+ algorithm.get_name()+"-"+problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name()+"-"+problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
