from math import cos, sin, acos, pi, sqrt, atan
import random

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution

"""
.. module:: unconstrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for single-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class OneMax(BinaryProblem):

    def __init__(self, number_of_bits: int = 256):
        super(OneMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        counter_of_ones = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1

        solution.objectives[0] = -1.0 * counter_of_ones

        return solution



    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def get_name(self) -> str:
        return 'OneMax'

class Linear(FloatProblem):
    def __init__(self, number_of_variables: int=9):
        super(Linear, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 8
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
         distance = 0
         S = solution.variables

         for i in range(0,9):
             distance += math.sqrt((S[2*i+2] - S[2*i])**2+(S[2*i+3]-S[2*i+1])**2)

         solution.objectives[0] = distance
         self.evaluate_constraints(solution)

         return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables
        e = 0.01

        constraints[0] = -S[0] - e
        constraints[1] = S[0] - e
        constraints[2] = -S[1] - e
        constraints[3] = S[1] - e
        constraints[4] = 5 -S[18] -e
        constraints[5] = -5 + S[18] - e
        constraints[6] = 5 - S[19] - e
        constraints[7] = -5 + S[19] - e


        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear'

class Linear2(FloatProblem):

    def __init__(self, number_of_variables: int=10):
        super(Linear2, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 12
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [5 for _ in range(number_of_variables)]



    def evaluate(self, solution: FloatSolution) -> FloatSolution:



         distance = 0
         S = solution.variables

         for i in range(0,5):
             distance += (math.sqrt((S[2 * i + 2] - S[2 * i]) ** 2)) / math.cos(
                 math.atan(
                     (math.sqrt((S[2 * i + 3] - S[2 * i + 1]) ** 2)) / (math.sqrt((S[2 * i + 2] - S[2 * i]) ** 2))))


         solution.objectives[0] = distance
         self.evaluate_constraints(solution)


         return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables

        e = 0.01

        constraints[0] = -S[0] - e
        constraints[1] = S[0] - e
        constraints[2] = -S[1] - e
        constraints[3] = S[1] - e
        constraints[4] = 1 -S[2] - e
        constraints[5] = -1 + S[2] - e
        constraints[6] = 2 -S[3] - e
        constraints[7] = -2 +S[3] - e
        constraints[8] = 5 -S[10] -e
        constraints[9] = -5 + S[10] - e
        constraints[10] = 5 - S[11] - e
        constraints[11] = -5 + S[11] - e





        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear2'


class Linear3(FloatProblem):

    def __init__(self, number_of_variables: int=10):
        super(Linear3, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 12
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0.25 for _ in range(number_of_variables)]
        self.upper_bound = [3 for _ in range(number_of_variables)]



    def evaluate(self, solution: FloatSolution) -> FloatSolution:
         Px = []
         Py = []
         Px.append(0)
         Py.append(0)

         distance = 0
         S = solution.variables
         del_t = 1
         for i in range(0,5):
             distance += math.sqrt((S[2 * i]*math.cos(S[2*i+1])-Px[i])**2 +
                                   (S[2 * i]*math.sin(S[2*i+1])-Py[i])**2)
             print(distance)
             x = Px[i] + math.sqrt((S[2*i]*math.cos(S[2*i+1]))**2)
             y = Py[i] + math.sqrt((S[2*i]*math.sin(S[2*i+1]))**2)


             Px.append(x)
             Py.append(y)

         for i in range(len(Px)):
             print(Px[i], Py[i])

         solution.objectives[0] = distance
         self.evaluate_constraints(solution)


         return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables

        e = 0.01

        constraints[0] = math.sqrt(2) -S[0] -e
        constraints[1] = -math.sqrt(2) + S[0] - e
        constraints[2] = 45*math.pi/180 - S[1] - e
        constraints[3] = -45*math.pi/180 + S[1] - e
        constraints[4] = math.sqrt(2) -S[2] -e
        constraints[5] = -math.sqrt(2) + S[2] - e
        constraints[6] = 45*math.pi/180 - S[3] - e
        constraints[7] = -45*math.pi/180 + S[3] - e
        constraints[8] = 5 + e - S[10]*math.cos(S[11])
        constraints[9] = -5 -e + S[10]*math.cos(S[11])
        constraints[10] = 5 + e - S[10]*math.sin(S[11])
        constraints[11] = -5 -e + S[10]*math.sin(S[11])

        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear3'

class Sphere(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

        return solution

    def get_name(self) -> str:
        return 'Sphere'


class Linear4(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Linear4, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        R_distance = []
        distance = 0
        S = solution.variables
        del_t = 1
        x = []
        y = []
        x.append(0)
        y.append(0)
        R_diviation = 0

        for i in range(0, 4):
            distance += S[2 * i] * del_t
            A =x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B =y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])
            x.append(A)
            y.append(B)

        R = sqrt((5-x[-1])**2 + (5-y[-1])**2)

        print(R)

        S[9] = acos(sqrt((5 - x[-1])**2) / R)

        xi = x[-1] + R * cos(S[9])
        yi = y[-1] + R * sin(S[9])
        x.append(xi)
        y.append(yi)

        for i in range(len(x)):
            print(x[i], y[i])



        for i in range(0, 5):
            R_mean = (distance + R) / 5

            R_diviation += sqrt((S[2*i]*del_t - R_mean)**2)




        distance = distance + R + R_diviation
        print(distance)



        solution.objectives[0] = distance
        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables
        del_t = 1

        e = 0.001

        # constraints[0] = math.sqrt(2) -S[0]*(del_t) + e
        # constraints[1] = -math.sqrt(2) + S[0]*(del_t) + e
        # constraints[2] = 45*math.pi/180 - S[1] + e
        # constraints[3] = -45*math.pi/180 + S[1] + e
        # constraints[4] = 5 + e - (S[0]*del_t*math.cos(S[1])+S[2]*del_t*math.cos(S[3])+S[4]*del_t*math.cos(S[5])+S[6]*del_t*math.cos(S[7])+S[8]*del_t*math.cos(S[9]))
        # constraints[5] = -5 + e + (S[0]*del_t*math.cos(S[1])+S[2]*del_t*math.cos(S[3])+S[4]*del_t*math.cos(S[5])+S[6]*del_t*math.cos(S[7])+S[8]*del_t*math.cos(S[9]))
        # constraints[6] = 5 + e - (S[0]*del_t*math.sin(S[1])+S[2]*del_t*math.sin(S[3])+S[4]*del_t*math.sin(S[5])+S[6]*del_t*math.sin(S[7])+S[8]*del_t*math.sin(S[9]))
        # constraints[7] = -5 + e + (S[0]*del_t*math.sin(S[1])+S[2]*del_t*math.sin(S[3])+S[4]*del_t*math.sin(S[5])+S[6]*del_t*math.sin(S[7])+S[8]*del_t*math.sin(S[9]))

        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear4'


class Rastrigin(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Rastrigin'


class SubsetSum(BinaryProblem):

    def __init__(self, C: int, W: list):
        """ The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.

        :param C: Large integer.
        :param W: Set of non-negative integers."""
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ['Sum']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_sum = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_sum += self.W[index]

        if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

        solution.objectives[0] = -1.0 * total_sum

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'Subset Sum'
