import math
from deap import gp

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def protected_division(x, y):
    if abs(y) < 1e-6:
        return 1.0
    return x / y

def sin(x):
    return math.sin(x)

def cos(x):
    return math.cos(x)

def if_then_else(condition_val1, condition_val2, val_if_true, val_if_false):
    if condition_val1 < condition_val2:
        return val_if_true
    else:
        return val_if_false


# pset.addPrimitive(add, 2, name="add")
# pset.addPrimitive(subtract, 2, name="sub")
# pset.addPrimitive(multiply, 2, name="mul")
# pset.addPrimitive(protected_division, 2, name="pdiv")
# pset.addPrimitive(sin, 1, name="sin")
# pset.addPrimitive(cos, 1, name="cos")
# pset.addPrimitive(if_then_else, 4, name="if")