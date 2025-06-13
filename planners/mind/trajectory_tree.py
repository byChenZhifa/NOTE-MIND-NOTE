import numpy as np
import theano.tensor as T
from common.geometry import get_point_mean_distances
from planners.basic.tree import Tree, Node  # 基础树结构
from planners.ilqr.solver import iLQR  # iLQR求解器
from planners.ilqr.dynamics import AutoDiffDynamics  # 自动微分动力学模型
from planners.ilqr.cost import TreeCost  # 树结构代价函数
from planners.ilqr.utils import gen_dist_field  # 生成距离场工具
from planners.ilqr.potential import ControlPotential, StatePotential, StateConstraint, PotentialField  # 各类势能函数


class TrajectoryTreeOptimizer:
    """
    - 轨迹树优化器(TrajectoryTreeOptimizer):
        1. 基于场景树(多概率分支)生成轨迹树
        2. 使用iLQR算法求解最优轨迹
        3. 支持预热初始化和精细优化两阶段求解

    - 核心方法:
        1. init_warm_start_cost_tree(): 构建简化代价树（仅目标跟踪）
        2. init_cost_tree(): 构建完整代价树（含目标跟踪+障碍规避）
        3. warm_start_solve(): 预热优化（生成初始解）
        4. solve(): 主优化（生成精细解）

    - 关键技术特点:
        1. 概率感知代价函数：场景概率加权各节点代价
        2. 距离场技术：将目标车道和障碍物表示为距离场
        3. 势能函数组合：
            PotentialField：空间位置代价
            StatePotential：状态偏差代价
            StateConstraint：状态边界约束
            ControlPotential：控制平滑性约束

        自行车模型：使用自动微分实现精确动力学
    - 优化流程:


    - 应用场景:
        自动驾驶轨迹规划
        考虑多交通参与者行为的不确定性
        需要满足动力学约束的轨迹优化
        多目标权衡（效率+安全+舒适性）
    """

    def __init__(self, config=None):
        """轨迹树优化器初始化"""
        self.config = config  # 配置参数对象
        # 创建动力学模型(自行车模型)，时间间隔为dt，轮距为2.5
        self.ilqr = iLQR(self._get_dynamic_model(self.config.dt, 2.5))
        self.cost_tree = None  # 用于存储代价函数的树结构
        self.debug = None  # 调试信息占位符

    def init_warm_start_cost_tree(self, scen_tree: Tree, init_state, init_ctrl, target_lane, target_vel):
        """初始化预热阶段的代价函数树（简化版）"""
        # 获取初始状态（包含位置、速度、朝向、加速度和转向角）
        x0 = self._get_init_state(init_state, init_ctrl)
        # 从配置获取网格分辨率和平滑网格尺寸
        res = self.config.w_opt_cfg["smooth_grid_res"]
        grid_size = self.config.w_opt_cfg["smooth_grid_size"]
        # 生成目标车道的距离场（欧氏距离）
        offsets, xx, yy, dist_field = gen_dist_field(x0, target_lane, grid_size, res)
        quad_dist_field = dist_field**2  # 平方距离（便于二次代价计算）

        # 创建代价树
        cost_tree = Tree()
        cost_tree.add_node(Node(-1, None, x0))  # 添加根节点（初始状态）
        # DFS to convert scenario tree to trajectory tree maintain the last traj node index of each scenario node
        # 用于记录每个场景节点对应的最后一个轨迹节点索引
        last_traj_node_index = {}
        queue = [scen_tree.get_root()]  # 初始化队列（场景树的根节点）

        # 场景树广度优先遍历（BFS）
        while queue:
            cur_node = queue.pop()
            # [prob] [ego + exo, N, state_dim] [ego + exo, N, state_dim, state_dim]
            prob, trajs, covs, tgt_pts = cur_node.data
            last_index = last_traj_node_index[cur_node.parent_key] if cur_node.parent_key is not None else -1
            duration = trajs.shape[1]
            for i in range(duration):
                if i % 2 == 1:
                    continue
                cur_index = len(cost_tree.nodes) - 1
                quad_cost_field = self.config.w_opt_cfg["w_tgt"] * prob * quad_dist_field
                pot_field = PotentialField(offsets, res, xx, yy, quad_cost_field)
                state_pot = StatePotential(self.config.w_opt_cfg["w_des_state"] * prob, np.array([0, 0, target_vel, 0.0, 0.0, 0.0]))
                state_con = StateConstraint(
                    self.config.w_opt_cfg["w_state_con"] * prob,
                    self.config.w_opt_cfg["state_lower_bound"],
                    self.config.w_opt_cfg["state_upper_bound"],
                )
                ctrl_pot = ControlPotential(self.config.w_opt_cfg["w_ctrl"] * prob)

                cost_tree.add_node(Node(cur_index, last_index, [[pot_field, state_pot, state_con], [ctrl_pot]]))
                last_index = cur_index
            last_traj_node_index[cur_node.key] = len(cost_tree.nodes) - 2
            for child_key in cur_node.children_keys:
                queue.append(scen_tree.get_node(child_key))

        self.cost_tree = TreeCost(cost_tree, self.config.state_size, self.config.action_size)

    def init_cost_tree(self, scen_tree: Tree, init_state, init_ctrl, target_lane, target_vel):
        """初始化完整代价函数树（考虑障碍物和自身安全边界）"""
        # 获取初始状态
        x0 = self._get_init_state(init_state, init_ctrl)
        res = self.config.opt_cfg["smooth_grid_res"]
        grid_size = self.config.opt_cfg["smooth_grid_size"]
        # 生成目标车道距离场
        offsets, xx, yy, dist_field = gen_dist_field(x0, target_lane, grid_size, res)
        centroids = np.vstack([xx.ravel(), yy.ravel()]).T  # 网格点坐标
        quad_dist_field = dist_field**2  # 平方距离场

        # 创建代价树
        cost_tree = Tree()
        cost_tree.add_node(Node(-1, None, x0))  # 根节点
        # DFS to convert scenario tree to trajectory tree maintain the last traj node index of each scenario node
        last_traj_node_index = {}  # 场景节点与轨迹节点的映射
        queue = [scen_tree.get_root()]  # BFS队列

        while len(queue) > 0:
            cur_node = queue.pop()
            # [prob] [ego + exo, N, state_dim] [ego + exo, N, state_dim, state_dim]
            prob, trajs, covs, tgt_pts = cur_node.data
            last_index = last_traj_node_index[cur_node.parent_key] if cur_node.parent_key is not None else -1

            duration = trajs.shape[1]  # 当前场景时长
            for i in range(duration):
                if i % 2 == 1:  # 跳步采样
                    continue
                cur_index = len(cost_tree.nodes) - 1
                cov_dist_field = dist_field * 0.0  # 初始化碰撞代价场

                # 自车安全场计算（扩展协方差边界）
                ego_mean = trajs[0, i]  # 自车位置均值
                # 自车协方差加上安全边界
                ego_cov = covs[0, i] + self.config.opt_cfg["w_ego_cov_offset"]
                # 计算到安全边界的有符号距离
                ego_dist_field = (get_point_mean_distances(centroids, ego_mean) - ego_cov).reshape(cov_dist_field.shape)
                ego_dist_field = np.maximum(ego_dist_field, 0.0)  # 转为非负距离

                # import matplotlib.pyplot as plt
                # _, ax = plt.subplots(figsize=(12, 12))
                # c = ax.pcolormesh(xx_discrete, yy_discrete, ego_dis_field, cmap='viridis', shading='auto')
                # plt.colorbar(c, ax=ax)
                # plt.show()

                # 计算他车障碍代价场
                for exo_idx in range(1, trajs.shape[0]):
                    exo_mean = trajs[exo_idx, i]  # 他车位置均值
                    # 他车协方差加上安全边界
                    exo_cov = covs[exo_idx, i] + self.config.opt_cfg["w_exo_cov_offset"]
                    # 计算到安全边界的有符号距离（负数表示内部）
                    exo_dis_field = (exo_cov - get_point_mean_distances(centroids, exo_mean)).reshape(cov_dist_field.shape)
                    exo_dis_field = np.maximum(exo_dis_field, 0.0)
                    # 在障碍物区域内添加基础代价偏移
                    exo_dis_field[exo_dis_field > 0] += self.config.opt_cfg["w_exo_cost_offset"]
                    cov_dist_field += exo_dis_field  # 累加他车代价
                pass

                # 组合总代价场 (目标跟踪 + 障碍规避 + 自车安全)
                quad_cost_field = (
                    self.config.opt_cfg["w_tgt"] * prob * quad_dist_field
                    + self.config.opt_cfg["w_exo"] * cov_dist_field
                    + self.config.opt_cfg["w_ego"] * ego_dist_field
                )

                # 创建各类势能
                pot_field = PotentialField(offsets, res, xx, yy, quad_cost_field)

                state_pot = StatePotential(self.config.opt_cfg["w_des_state"] * prob, np.array([0, 0, target_vel, 0.0, 0.0, 0.0]))

                state_con = StateConstraint(
                    self.config.opt_cfg["w_state_con"] * prob, self.config.opt_cfg["state_lower_bound"], self.config.opt_cfg["state_upper_bound"]
                )
                ctrl_pot = ControlPotential(self.config.opt_cfg["w_ctrl"] * prob)

                # 添加节点到代价树
                cost_tree.add_node(Node(cur_index, last_index, [[pot_field, state_pot, state_con], [ctrl_pot]]))
                last_index = cur_index
            pass

            # 记录当前场景节点的最后轨迹节点
            last_traj_node_index[cur_node.key] = len(cost_tree.nodes) - 2
            # 处理子节点
            for child_key in cur_node.children_keys:
                queue.append(scen_tree.get_node(child_key))

        # 构建树结构代价函数
        self.cost_tree = TreeCost(cost_tree, self.config.state_size, self.config.action_size)

    def warm_start_solve(self, us_init=None):
        """执行预热优化（生成粗略初始解）"""
        # 默认初始化控制量为零
        if us_init is None:
            us_init = np.zeros((self.cost_tree.tree.size() - 1, self.config.action_size))

        # 调用iLQR求解器进行优化
        xs, us = self.ilqr.fit(us_init, self.cost_tree)
        return xs, us  # 返回状态序列和控制序列

    def solve(self, us_init=None):
        """求解最优轨迹树;主求解函数（生成精细解）
        输入：
            us_init: 初始控制序列
        返回：
            优化后的轨迹树
        """
        # 初始化控制序列
        if us_init is None:
            us_init = np.zeros((self.cost_tree.tree.size() - 1, self.config.action_size))

        # 使用iLQR算法进行优化
        xs, us = self.ilqr.fit(us_init, self.cost_tree)

        # 构建返回的轨迹树结构
        # return traj tree
        traj_tree = Tree()
        for node in self.cost_tree.tree.nodes.values():
            if node.parent_key is None:  # 根节点处理
                # 根节点控制量设为零
                traj_tree.add_node(Node(node.key, None, [node.data, np.zeros(self.config.action_size)]))
            else:  # 普通节点
                # 节点数据包含状态和优化后的控制量
                traj_tree.add_node(Node(node.key, node.parent_key, [xs[node.key], us[node.key]]))
        return traj_tree  # 返回优化后的轨迹树

    def _get_init_state(self, init_state, init_ctrl):
        """构建扩展初始状态向量[x, y, v, θ, a, δ]"""
        return np.array(
            [
                init_state[0],  # x位置
                init_state[1],  # y位置
                init_state[2],  # 速度
                init_state[3],  # 航向角
                init_ctrl[0],  # 加速度
                init_ctrl[1],  # 转向角
            ]
        )

    def _get_dynamic_model(self, dt, wb):
        """构建自行车动力学模型（自动微分）"""
        # 状态变量定义: [x, y, v, q, a, θ]
        x_inputs = [
            T.dscalar("x"),  # x位置
            T.dscalar("y"),  # y位置
            T.dscalar("v"),  # 速度
            T.dscalar("q"),  # 航向角
            T.dscalar("a"),  # 加速度
            T.dscalar("theta"),  # 前轮转向角
        ]

        # 控制变量定义: [da, dθ]
        u_inputs = [
            T.dscalar("da"),
            T.dscalar("dtheta"),
        ]

        # 状态更新方程（离散时间自行车模型）
        f = T.stack(
            [
                # x更新: x + vcos(q)dt
                x_inputs[0] + x_inputs[2] * T.cos(x_inputs[3]) * dt,
                # y更新: y + vsin(q)dt
                x_inputs[1] + x_inputs[2] * T.sin(x_inputs[3]) * dt,
                # 速度更新: v + a*dt
                x_inputs[2] + x_inputs[4] * dt,
                # 航向角更新: q + (v/wb)tan(theta)dt
                x_inputs[3] + x_inputs[2] / wb * T.tan(x_inputs[5]) * dt,
                # 加速度更新: a + da*dt
                x_inputs[4] + u_inputs[0] * dt,
                # 转向角更新: theta + dtheta*dt
                x_inputs[5] + u_inputs[1] * dt,
            ]
        )

        return AutoDiffDynamics(f, x_inputs, u_inputs)
