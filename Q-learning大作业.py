import random

# 封装环境初始化
def initialize_environment():
   """
   初始化 Q-learning 的超参数和动态背包问题数据
  :return: 参数字典，包括 gamma、alpha、epsilon、episodes、
   weights、values、capacity 等
   """
   params = {
   "gamma": 0.9, # 折扣因子
   "alpha": 0.01, # 学习率
   "epsilon": 0.2, # 探索概率
   "episodes": 30000, # 最大训练轮次
   "weights": [32, 17, 5, 19, 14, 6, 25, 23, 24, 36], # 物品重量
   "values": [66, 33, 10, 39, 47, 4, 36, 60, 8, 20], # 物品价值
   "capacity": 44, # 背包容量
   }
   return params


class State:
    """
    表示动态背包问题的状态
    """

    def __init__(self, capacity, weights, values, current_index=0, items=None):
        """
        初始化状态
        :param capacity: 背包容量
        :param weights: 所有物品的重量列表
        :param values: 所有物品的价值列表
        :param current_index: 当前处理的物品索引
        :param items: 当前背包中的物品（列表形式存储物品索引）
        """
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.current_index = current_index
        self.items = items if items is not None else []

    def is_terminal(self):
        """
        判断当前状态是否为终止状态
        :return: 布尔值，表示是否为终止状态
        """
        return self.current_index >= len(self.weights)

    def get_valid_actions(self):
        """
        获取当前状态下的所有合法动作
        :return: 动作列表 ["add", "replace", "skip"]
        """
        # 如果当前索引超出范围，返回空动作列表
        if self.is_terminal():
            return []

        actions = ["skip"]  # "跳过"动作总是合法的
        current_weight = self.weights[self.current_index]

        # 判断是否可以"添加"
        if sum(self.weights[i] for i in self.items) + current_weight <= self.capacity:
            actions.append("add")

        # 判断是否可以"替换"
        for i in range(len(self.items)):
            if (
                    sum(self.weights[j] for j in self.items)
                    - self.weights[self.items[i]]
                    + current_weight
                    <= self.capacity
            ):
                actions.append(f"replace_{i}")  # 用当前物品替换第 i 个物品

        return actions

    def transition(self, action):
        """
        执行动作，返回新的状态和即时奖励
        :param action: 动作
        :return: (新状态, 奖励)
        """
        # 如果已经是终止状态，直接返回当前状态和 0 奖励
        if self.is_terminal():
            return self, 0

        if action == "skip":
            return (
                State(
                    self.capacity,
                    self.weights,
                    self.values,
                    self.current_index + 1,
                    self.items,
                ),
                0,  # 跳过的即时奖励为 0
            )

        elif action == "add":
            new_items = self.items + [self.current_index]
            reward = self.values[self.current_index]
            return (
                State(
                    self.capacity,
                    self.weights,
                    self.values,
                    self.current_index + 1,
                    new_items,
                ),
                reward,
            )

        elif action.startswith("replace"):
            replace_index = int(action.split("_")[1])
            new_items = self.items[:]
            new_items[replace_index] = self.current_index
            reward = (
                    self.values[self.current_index]
                    - self.values[self.items[replace_index]]
            )
            return (
                State(
                    self.capacity,
                    self.weights,
                    self.values,
                    self.current_index + 1,
                    new_items,
                ),
                reward,
            )

    def __hash__(self):
        """
        实现哈希函数，用于 Q 值表索引
        """
        return hash((tuple(self.items), self.current_index))

    def __eq__(self, other):
        """
        实现状态比较逻辑
        """
        return (
                self.items == other.items
                and self.current_index == other.current_index
        )

# 选择动作
def choose_action(state, Q, epsilon):
    """
    基于 epsilon-greedy 策略选择动作
    :param state: 当前状态
    :param Q: Q 值表
    :param epsilon: 探索概率
    :return: 选择的动作
    """
    valid_actions = state.get_valid_actions()
    if random.random() < epsilon:
        # 探索：随机选择一个合法动作
        return random.choice(valid_actions)
    else:
        # 利用：选择 Q 值最大的动作
        state_key = hash(state)
        if state_key not in Q:
            Q[state_key] = {action: 0 for action in valid_actions}
        return max(valid_actions, key=lambda action: Q[state_key].get(action, 0))

# Q-learning 算法
def q_learning(params):
    """
    Q-learning 算法实现
    """
    Q = {}  # 初始化 Q 值表

    for episode in range(params["episodes"]):
        # 初始化状态
        state = State(params["capacity"], params["weights"], params["values"])

        while not state.is_terminal():
            state_key = hash(state)

            # 选择动作
            action = choose_action(state, Q, params["epsilon"])
            if state_key not in Q:
                Q[state_key] = {action: 0 for action in state.get_valid_actions()}

            # 执行动作，计算新状态和即时奖励
            new_state, reward = state.transition(action)
            new_state_key = hash(new_state)

            # 初始化 Q 表
            if new_state_key not in Q:
                valid_actions = new_state.get_valid_actions()
                if valid_actions:
                    Q[new_state_key] = {a: 0 for a in valid_actions}
                else:
                    Q[new_state_key] = {}  # 如果没有合法动作，初始化为空字典

            # 计算 max_next_q
            max_next_q = max(Q[new_state_key].values()) if Q[new_state_key] else 0

            # 更新 Q 值
            Q[state_key][action] += params["alpha"] * (
                    reward + params["gamma"] * max_next_q - Q[state_key][action]
            )

            # 更新为新状态
            state = new_state

            # **新增检查：如果状态为终止状态，退出循环**
            if state.is_terminal():
                break

    return Q

# 提取最优策略
def extract_policy(Q, params):
    """
    从 Q 表中提取最优策略
    """
    state = State(params["capacity"], params["weights"], params["values"])
    policy = []
    total_value = 0
    used_capacity = 0

    while not state.is_terminal():
        state_key = hash(state)
        if state_key not in Q or not Q[state_key]:
            break

        # 提取当前状态下的最优动作
        best_action = max(Q[state_key], key=Q[state_key].get)
        policy.append(best_action)

        # 执行动作，更新状态和价值
        new_state, reward = state.transition(best_action)
        total_value += reward
        used_capacity = sum(state.weights[i] for i in state.items)
        state = new_state

    return policy, total_value, used_capacity

# 主函数
if __name__ == "__main__":
    # 初始化环境
    params = initialize_environment()

    # 训练 Q-learning
    Q = q_learning(params)

    # 输出结果
    print("最优策略（动作序列）:")
    optimal_policy, total_value, total_capacity_used = extract_policy(Q, params)
    print(optimal_policy)
    print("最大价值:")
    print(total_value)
    print("总占用容量:")
    print(total_capacity_used)