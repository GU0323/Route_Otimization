import matplotlib.pyplot as plt
import matplotlib.patches as patches

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Linear4
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from math import cos, sin


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
    print(front)

    # Save results to file
    print_function_values_to_file(front, 'FUN.'+ algorithm.get_name()+"-"+problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name()+"-"+problem.get_name())
    file_open = open('VAR.' + algorithm.get_name()+"-"+problem.get_name(), 'r', encoding='utf-8')
    line = file_open.readlines()
    word = line[-1].split(" ")
    x = []
    y = []
    x.append(0)
    y.append(0)
    del_t = 1
    for i in range(int(21)):
        xi = x[i] + float(word[2*i])*del_t*cos(float(word[2*i+1]))
        yi = y[i] + float(word[2*i])*del_t*sin(float(word[2*i+1]))
        x.append(xi)
        y.append(yi)

    plt.plot(x, y)
    shp = patches.Circle((7,7), radius = 4, color='g')
    plt.gca().add_patch(shp)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.show()

    file_open.close()

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
