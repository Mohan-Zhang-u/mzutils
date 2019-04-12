import math


def permutation(a,b):
    """
    Calculate the number of permutations from a choose b
    """
    if a > b:
        return int(math.factorial(a)/math.factorial(a-b))
    raise Exception("a should be less than b")


def binomial_coefficient(a, b):
    if a > b:
        return int(math.factorial(a)/math.factorial(a-b)/math.factorial(b))
    raise Exception("a should be less than b")

