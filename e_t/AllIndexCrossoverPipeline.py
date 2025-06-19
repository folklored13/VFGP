import random
from deap import gp


class AllIndexCrossover:
    def __init__(self, likelihood=0.9, max_depth=17):
        self.likelihood = likelihood
        self.max_depth = max_depth

    def __call__(self, ind1, ind2):
        if random.random() > self.likelihood:
            return ind1, ind2

        if len(ind1) != len(ind2):
            raise ValueError("Individuals must have same number of trees")

        new_ind1, new_ind2 = ind1.copy(), ind2.copy()

        for i in range(len(ind1)):
            tree1 = new_ind1[i]
            tree2 = new_ind2[i]

            # 选择交叉点
            point1 = self.select_node(tree1)
            point2 = self.select_node(tree2)

            # 验证交叉有效性
            if self.verify_points(point1, point2):
                # 执行子树交换
                slice1 = tree1.searchSubtree(point1)
                slice2 = tree2.searchSubtree(point2)
                tree1[slice1], tree2[slice2] = tree2[slice2], tree1[slice1]

        return new_ind1, new_ind2

    def select_node(self, tree):
        # 随机选择有效节点 跳过根节点
        nodes = [node for node in tree[1:] if node.valid]
        return random.choice(nodes) if nodes else None

    def verify_points(self, point1, point2):
        #验证交叉点兼容性
        if not point1 or not point2:
            return False

        # 检查深度兼容性
        depth1 = point1.depth
        depth2 = point2.depth
        if depth1 + depth2 > self.max_depth:
            return False

        # 检查类型兼容性
        return point1.ret == point2.ret