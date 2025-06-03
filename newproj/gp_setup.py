import operator
import numpy as np
from deap import base, creator, tools, gp

# --- 自定义原语 ---
def protectedDiv(left, right):
    try:
        # 使用 numpy 的除法，并处理无穷大和 NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            # 确保输入是浮点数
            fleft = float(left)
            fright = float(right)
            # 避免非常小的分母导致溢出或不精确
            if abs(fright) < 1e-9:
                 return 0.0 # 或者 1.0
            res = np.divide(fleft, fright)
            # 检查结果是否有限
            if not np.isfinite(res):
                return 0.0 # 或者根据需要返回其他值
            return res
    except (ZeroDivisionError, OverflowError, ValueError): # 添加 OverflowError 和 ValueError
        return 0.0 # 或者 1.0

def sin(x):
    try: return np.sin(float(x))
    except (ValueError, OverflowError): return 0.0 # 处理无效输入

def cos(x):
    try: return np.cos(float(x))
    except (ValueError, OverflowError): return 0.0 # 处理无效输入

def IF(a, b, c, d):
    try: return c if float(a) < float(b) else d
    except (ValueError, OverflowError): return d # 如果比较失败，返回 else 分支

def create_toolbox(n_features):
    """
    创建并配置 DEAP Toolbox。
    Args:
        n_features (int): 输入特征的数量。
    Returns:
        deap.base.Toolbox: 配置好的 Toolbox 实例。
    """
    pset = gp.PrimitiveSet("MAIN", arity=n_features)

    # 添加基本算术运算符
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)

    # 添加三角函数
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)

    # 添加条件操作符 (IF a < b THEN c ELSE d)
    pset.addPrimitive(IF, 4)

    # 重命名参数为 F0, F1, ...
    arg_names = {f"ARG{i}": f"F{i}" for i in range(n_features)}
    try:
        pset.renameArguments(**arg_names)
    except ValueError as e:
         print(f"Error renaming arguments (check n_features={n_features}): {e}")
         print(f"Provided arg names: {list(arg_names.keys())}")
         # Fallback or raise error
         pass


    # --- 定义 Fitness 和 Individual ---
    # 检查是否已创建，防止重复创建引发错误
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    # --- 初始化 Toolbox ---
    toolbox = base.Toolbox()

    # 注册 GP 树生成器
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=config.TREE_MIN_DEPTH, max_=config.TREE_MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册编译函数
    toolbox.register("compile", gp.compile, pset=pset)

    # --- 注册遗传算子 ---
    # 注意：evaluate 需要在主循环中注册，因为它依赖特定 fold 的数据
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    # 变异表达式生成器
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2) # 用于变异的子树深度
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 添加树高限制防止膨胀
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_TREE_HEIGHT))

    return toolbox

# 导入 config 模块以访问参数
import config