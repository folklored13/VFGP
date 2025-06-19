import random
from deap import gp

# 全局变量
NUMBER_OF_FEATURES = 0
CURRENT_SAMPLE = None
RANDOM_GENERATOR = random.Random()

class F(gp.Terminal):
    def __init__(self, name="F"):
        super().__init__(name)
        self.index = -1 # 初始化索引

    def format(self): # DEAP 中用于生成字符串表示的方法
        if self.index == -1: # 在第一次格式化/评估前设置
            self.index = RANDOM_GENERATOR.randint(0, NUMBER_OF_FEATURES - 1)
        return f"F{self.index}"

    def eval(self):
        if CURRENT_SAMPLE is None:
            raise ValueError("CURRENT_SAMPLE is not set in gp_terminals")
        return CURRENT_SAMPLE.get_feature_of(self.index)

class Constant(gp.Terminal):
    def __init__(self, name="Const"):
        super().__init__(name)
        self.value = -1

    def format(self):
        if self.value == -1:
            self.value = RANDOM_GENERATOR.uniform(-1.0, 1.0)
        return f"{self.value:.2f}"

    def eval(self):
        return self.value

# 在 DEAP 中，你通常会定义一个 PrimitiveSet (pset)
# pset = gp.PrimitiveSet("MAIN", arity=0) # arity 0 表示没有输入参数给整个树
# pset.addTerminal(F, name="F")
# pset.addTerminal(Constant, name="Const")