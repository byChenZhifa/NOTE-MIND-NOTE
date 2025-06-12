import json
import numpy as np
import torch
import time
from importlib import import_module  # 动态模块导入
from common.geometry import project_point_on_polyline  # 几何投影工具
from planners.mind.scenario_tree import ScenarioTreeGenerator  # 场景树生成器
from planners.mind.trajectory_tree import TrajectoryTreeOptimizer  # 轨迹树优化器
from av2.datasets.motion_forecasting.data_schema import Track, ObjectState, TrackCategory, ObjectType


class MINDPlanner:
    """MIND规划器主类，负责场景树生成和轨迹优化"""

    def __init__(self, config_dir):
        """初始化规划器

        参数:
            config_dir: 配置文件路径
        """
        # 配置参数
        self.planner_cfg = None  # 规划器整体配置
        self.network_cfg = None  # 神经网络配置
        self.device = None  # 计算设备(CPU/GPU)
        self.network = None  # 神经网络模型
        self.scen_tree_gen = None  # 场景树生成器
        self.traj_tree_opt = None  # 轨迹树优化器
        self.obs_len = 50  # 历史观测长度(时间步)
        self.plan_len = 50  # 规划长度(时间步)
        self.agent_obs = {}  # 智能体观测数据字典
        self.state = None  # 自车当前状态
        self.ctrl = None  # 自车当前控制指令
        self.gt_tgt_lane = None  # 真实目标车道点
        self.last_ctrl_seq = []  # 上一步控制指令序列缓存

        # 初始化配置参数
        with open(config_dir, "r") as file:  # 打开配置文件
            self.planner_cfg = json.load(file)  # 加载JSON配置
        self.init_device()  # 初始化计算设备
        self.init_network()  # 加载神经网络模型
        self.init_scen_tree_gen()  # 初始化场景树生成器
        self.init_traj_tree_opt()  # 初始化轨迹树优化器

    def init_device(self):
        """根据配置选择计算设备 (CPU/GPU)"""
        # 检查是否启用CUDA且有可用GPU
        if self.planner_cfg["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)  # 使用第一个GPU设备
        else:
            self.device = torch.device("cpu")  # 回退到CPU

    def init_network(self):
        """加载预训练神经网络模型"""
        # 动态导入网络配置模块
        self.network_cfg = import_module(self.planner_cfg["network_config"]).NetCfg()
        # 获取网络配置字典
        net_cfg = self.network_cfg.get_net_cfg()
        # 解析网络定义文件路径和类名（格式: "file.path:ClassName"）
        net_file, net_name = net_cfg["network"].split(":")

        # 动态导入网络类
        self.network = getattr(import_module(net_file), net_name)(net_cfg, self.device)
        # 加载预训练权重
        ckpt = torch.load(self.planner_cfg["ckpt_path"], map_location=lambda storage, loc: storage)
        # 将权重加载到网络
        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()  # 设置为评估模式(禁用dropout等)

    def init_scen_tree_gen(self):
        """初始化场景树生成器"""
        # 动态导入场景树配置
        scen_tree_cfg = import_module(self.planner_cfg["planning_config"]).ScenTreeCfg()
        # 创建场景树生成器实例
        self.scen_tree_gen = ScenarioTreeGenerator(
            self.device, self.network, self.obs_len, self.plan_len, scen_tree_cfg  # 计算设备  # 预测模型  # 观测长度  # 规划长度  # 配置参数
        )

    def init_traj_tree_opt(self):
        """初始化轨迹树优化器"""
        # 动态导入轨迹树配置
        traj_tree_cfg = import_module(self.planner_cfg["planning_config"]).TrajTreeCfg()
        # 创建轨迹树优化器实例
        self.traj_tree_opt = TrajectoryTreeOptimizer(traj_tree_cfg)
        passss = 1  # 占位符（无实际作用）

    def to_object_state(self, agent):
        """将智能体状态转换为Argo Dataset的标准格式"""
        obj_state = ObjectState(
            True,  # 有效状态标志
            agent.timestep,  # 时间戳
            (agent.state[0], agent.state[1]),  # (x, y)位置
            agent.state[3],  # 航向角
            # (vx, vy)速度向量
            (agent.state[2] * np.cos(agent.state[3]), agent.state[2] * np.sin(agent.state[3])),
        )
        return obj_state

    def update_observation(self, lcl_smp):
        """更新智能体观测信息"""
        # 更新自车状态
        if "AV" not in self.agent_obs:  # 首次观察自车
            # 创建自车轨迹对象
            self.agent_obs["AV"] = Track(
                "AV",  # 轨迹ID
                [self.to_object_state(lcl_smp.ego_agent)],  # 状态序列
                lcl_smp.ego_agent.type,  # 智能体类型
                TrackCategory.FOCAL_TRACK,  # 主轨迹类别
            )
        else:
            # 添加新状态到现有轨迹
            self.agent_obs["AV"].object_states.append(self.to_object_state(lcl_smp.ego_agent))

        # 更新其他交通参与者状态
        updated_agent_ids = ["AV"]  # 已更新智能体ID集合
        for agent in lcl_smp.exo_agents:  # 遍历所有外部智能体
            if agent.id not in self.agent_obs:  # 首次观察该智能体
                # 创建新轨迹
                self.agent_obs[agent.id] = Track(
                    agent.id, [self.to_object_state(agent)], agent.type, TrackCategory.TRACK_FRAGMENT  # 智能体ID  # 状态序列  # 类型  # 轨迹片段类别
                )
            else:
                # 添加新状态到现有轨迹
                self.agent_obs[agent.id].object_states.append(self.to_object_state(agent))
            updated_agent_ids.append(agent.id)  # 添加到已更新集合

        # 为未观察到的智能体添加虚拟状态
        for agent in self.agent_obs.values():  # 遍历所有已知智能体
            if agent.track_id not in updated_agent_ids:  # 该智能体本次未更新
                # 添加无效状态（复制最后已知状态）
                agent.object_states.append(
                    ObjectState(
                        False,  # 无效标志
                        agent.object_states[-1].timestep,  # 保持相同时间戳
                        agent.object_states[-1].position,  # 相同位置
                        agent.object_states[-1].heading,  # 相同航向
                        agent.object_states[-1].velocity,  # 相同速度
                    )
                )

        # 保持历史长度不超过观测窗口大小
        for agent in self.agent_obs.values():
            if len(agent.object_states) > self.obs_len:
                # 移除最旧的状态
                agent.object_states.pop(0)

    def update_state_ctrl(self, state, ctrl):
        """更新自车当前状态和控制指令"""
        self.state = state  # 当前状态 [x, y, v, θ]
        self.ctrl = ctrl  # 当前控制 [加速度, 转向角]

    def update_target_lane(self, gt_tgt_lane):
        """更新真实目标车道点"""
        self.gt_tgt_lane = gt_tgt_lane

    def plan(self, lcl_smp):
        """规划主流程
        输入：
            lcl_smp: LocalSemanticMap对象，包含局部语义信息
        返回：
            (是否成功, 最优控制指令, [场景树,轨迹树])
        """
        t0 = time.time()  # 记录起始时间
        # 重置规划器状态
        self.scen_tree_gen.reset()  # 重置场景树生成器

        # 目标车道重采样（减小点间距）
        resample_target_lane, resample_target_lane_info = self.resample_target_lane(lcl_smp)
        self.scen_tree_gen.set_target_lane(resample_target_lane, resample_target_lane_info)

        # 1. ========= 生成候选场景树 =========
        scen_trees = self.scen_tree_gen.branch_aime(lcl_smp, self.agent_obs)

        # 检查场景树生成结果
        if len(scen_trees) < 0:  # 如果未生成任何场景树
            return False, None, None  # 规划失败

        # 2. ========= 轨迹优化与评估（为每个场景树分支） =========
        traj_trees = []  # 轨迹树集合
        debug_info = []  # 调试信息集合
        for scen_tree in scen_trees:  # 遍历每个场景树
            # 执行轨迹优化
            traj_tree, debug = self.get_traj_tree(scen_tree, lcl_smp)
            traj_trees.append(traj_tree)  # 存储轨迹树
            debug_info.append(debug)  # 存储调试信息

        # use multi-threading to speed up
        # n_proc = len(scen_trees)
        # traj_trees = Parallel(n_jobs=n_proc)(
        #     delayed(self.get_traj_tree)(scen_tree, lcl_smp) for scen_tree in scen_trees)

        # 3. ========= 多目标决策选择最优轨迹 =========
        best_traj_idx = None  # 最优轨迹索引
        min_cost = np.inf  # 最小代价（初始为无穷大）
        for idx, traj_tree in enumerate(traj_trees):  # 遍历所有轨迹树
            cost = self.evaluate_traj_tree(lcl_smp, traj_tree)  # 评估轨迹树代价
            if cost < min_cost:  # 找到更小代价
                min_cost = cost  # 更新最小代价
                best_traj_idx = idx  # 更新最优索引

        # 4. ========= 提取最优控制指令 =========
        opt_traj_tree = traj_trees[best_traj_idx]  # 获取最优轨迹树
        # 获取根节点的第一个子节点
        next_node = opt_traj_tree.get_node(opt_traj_tree.get_root().children_keys[0])
        # 提取控制指令（最后两个元素：加速度和转向角）
        ret_ctrl = next_node.data[0][-2:]

        # 返回结果：成功标志、控制指令、场景树和轨迹树
        return True, ret_ctrl, [[scen_trees[best_traj_idx]], [traj_trees[best_traj_idx]]]

    def resample_target_lane(self, lcl_smp):
        """目标车道重采样
        输入：
            lcl_smp: LocalSemanticMap对象
        返回：
            (resample_target_lane, resample_target_lane_info) - 重采样后的车道坐标及属性
        """
        # 初始化结果容器
        resample_target_lane = []  # 重采样车道点
        # 车道属性（6个特征数组）
        resample_target_lane_info = [[] for _ in range(6)]

        # 遍历目标车道的每个连续点对
        for i in range(len(lcl_smp.target_lane) - 1):
            # 当前车道段
            lane_segment = lcl_smp.target_lane[i : i + 2]
            # 计算段长度
            lane_segment_len = np.linalg.norm(lane_segment[0] - lane_segment[1])
            # 计算采样点数（每米一个点）
            num_sample = int(np.ceil(lane_segment_len / 1.0))

            # 线性插值生成新点
            for j in range(num_sample):
                alpha = j / num_sample  # 插值比例
                # 计算新点位置
                new_point = lane_segment[0] + alpha * (lane_segment[1] - lane_segment[0])
                resample_target_lane.append(new_point)

                # 同步车道属性（通过线性插值）
                for k, info in enumerate(lcl_smp.target_lane_info):
                    resample_target_lane_info[k].append(info[i])

        # 添加最后一个点（不变）
        resample_target_lane.append(lcl_smp.target_lane[-1])
        for k, info in enumerate(lcl_smp.target_lane_info):
            resample_target_lane_info[k].append(info[-1])

        # 转换为numpy数组
        resample_target_lane = np.array(resample_target_lane)
        for i in range(len(resample_target_lane_info)):
            resample_target_lane_info[i] = np.array(resample_target_lane_info[i])

        return resample_target_lane, resample_target_lane_info

    def get_traj_tree(self, scen_tree, lcl_smp):
        """为给定场景树生成优化轨迹树

        参数:
            scen_tree: 场景树
            lcl_smp: 局部语义地图

        返回:
            (轨迹树, 调试信息)
        """
        # 初始化轨迹树优化器的成本树（预热启动）
        self.traj_tree_opt.init_warm_start_cost_tree(
            scen_tree,  # 场景树
            self.state,  # 当前状态
            self.ctrl,  # 当前控制
            self.gt_tgt_lane,  # 目标车道
            lcl_smp.target_velocity,  # 目标速度
        )
        # 预热求解（获取初始控制序列）
        xs, us = self.traj_tree_opt.warm_start_solve()

        # 初始化正式成本树
        self.traj_tree_opt.init_cost_tree(
            scen_tree,  # 场景树
            self.state,  # 当前状态
            self.ctrl,  # 当前控制
            self.gt_tgt_lane,  # 目标车道
            lcl_smp.target_velocity,  # 目标速度
        )
        # 使用预热结果求解轨迹树
        return self.traj_tree_opt.solve(us), self.traj_tree_opt.debug

    def evaluate_traj_tree(self, lcl_smp, traj_tree):
        """轨迹树评估函数
        输入：
            lcl_smp: 局部语义地图
            traj_tree: 待评估的轨迹树
        返回：
            综合成本值 (越小越好)

        comfort_cost = 0.0  # 舒适性成本
        efficiency_cost = 0.0  # 效率成本
        target_cost = 0.0  # 目标追踪成本
        """
        # we use cost function here, instead of the reward function in the paper, but reward functions can work as well
        # simplified cost function
        # 初始化各成本分量权重
        comfort_acc_weight = 0.1  # 加速度舒适性权重
        comfort_str_weight = 5.0  # 转向舒适性权重
        comfort_cost = 0.0  # 舒适性总成本
        efficiency_weight = 0.01  # 效率权重
        efficiency_cost = 0.0  # 效率成本
        target_weight = 0.01  # 目标追踪权重
        target_cost = 0.0  # 目标追踪成本

        n_nodes = len(traj_tree.nodes)  # 轨迹树节点总数

        # 遍历轨迹树所有节点
        for node in traj_tree.nodes.values():
            state = node.data[0]  # 状态向量 [x, y, v, θ, a, δ]
            ctrl = node.data[1]  # 控制向量 [a, δ]

            # 舒适性计算 (加速度和转向惩罚)
            comfort_cost += comfort_acc_weight * ctrl[0] ** 2  # 加速度成本
            comfort_cost += comfort_str_weight * ctrl[1] ** 2  # 转向成本

            # 效率计算 (与目标速度的偏差)
            efficiency_cost += efficiency_weight * (lcl_smp.target_velocity - state[2]) ** 2

            # 目标追踪计算 (到目标车道的距离)
            target_cost += target_weight * self.get_dist_to_target_lane(lcl_smp, state)

        # 计算各节点平均成本
        return (comfort_cost + efficiency_cost + target_cost) / n_nodes

    def get_dist_to_target_lane(self, lcl_smp, state):
        """计算状态点到目标车道的最近距离"""
        #  project the state to the target lane
        proj_state, _, _ = project_point_on_polyline(
            state[:2],  # 位置坐标 (x, y)
            lcl_smp.target_lane,  # 目标车道点序列
        )
        #  get the distance
        dist = np.linalg.norm(proj_state - state[:2])
        return dist

    def get_interpolated_state(self, tree, timestep):
        """根据时间戳在轨迹树中插值状态

        参数:
            tree: 轨迹树
            timestep: 目标时间戳

        返回:
            (插值状态, 控制指令)
        """
        root_node = tree.get_node(0)  # 获取根节点
        if timestep < root_node.data.t:  # 时间戳在根节点之前
            return root_node.data.state, root_node.data.ctrl
        else:
            node = root_node
            # 查找第一个时间超过目标的时间节点
            while node.data.t <= timestep:
                node = tree.get_node(node.children_keys[0])
            # 获取前一节点（父节点）
            prev_node = tree.get_node(node.parent_key)
            # 前后节点状态
            prev_state = prev_node.data.state
            next_state = node.data.state
            # 前后节点时间戳
            prev_time = prev_node.data.t
            next_time = node.data.t
            # 计算插值系数
            alpha = (timestep - prev_time) / (next_time - prev_time)
            # 线性插值
            interp_state = prev_state + alpha * (next_state - prev_state)
            return interp_state, node.data.ctrl
