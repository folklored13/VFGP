from deap import algorithms, base, creator, tools, gp
import operator

def setup_gp_primitives(feature_dim):
    """配置GP函数集和终端集"""
    pset = gp.PrimitiveSet("MAIN", arity=0)

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.truediv, 2)

    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    #论文中的if
    pset.addPrimitive(np.maximum, 2, name='if_positive')
    #终端节点：随机选择特征
    pset.addEphemeralConstant("rand_feat", lambda: np.random.randint(0, feature_dim))
    return pset


def create_gp_toolbox(pset, X_train, y_train):
    """创建DEAP工具箱"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # 注册适应度评估函数
    toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    return toolbox


def evaluate_individual(individual, X_train, y_train):
    """适应度函数：集成分类的F1分数"""
    try:
        func = toolbox.compile(expr=individual)
        # 特征转换
        X_trans = np.array([[func(feat) for feat in sample] for sample in X_train])
        # 训练集成分类器
        eclf = VotingClassifier([('svm', SVC()), ('dt', DecisionTreeClassifier()), ('rf', RandomForestClassifier())])
        eclf.fit(X_trans, y_train)
        y_pred = eclf.predict(X_trans)
        return (f1_score(y_train, y_pred, average='weighted'),)
    except:
        return (0.0,)  # 处理无效个体