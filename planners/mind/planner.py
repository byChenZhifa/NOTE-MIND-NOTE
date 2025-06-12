import json
import numpy as np
import torch
import time
from importlib import import_module
from common.geometry import project_point_on_polyline
from planners.mind.scenario_tree import ScenarioTreeGenerator
from planners.mind.trajectory_tree import TrajectoryTreeOptimizer
from av2.datasets.motion_forecasting.data_schema import Track, ObjectState, TrackCategory, ObjectType


class MINDPlanner:
    def __init__(self, config_dir):
        # 初始化配置参数

        self.planner_cfg = None
        self.network_cfg = None
        self.device = None
        self.network = None
        self.scen_tree_gen = None
        self.traj_tree_opt = None
        self.obs_len = 50
        self.plan_len = 50
        self.agent_obs = {}
        self.state = None
        self.ctrl = None
        self.gt_tgt_lane = None
        self.last_ctrl_seq = []

        # 初始化配置参数
        with open(config_dir, "r") as file:
            self.planner_cfg = json.load(file)
        self.init_device()  # 初始化计算设备
        self.init_network()  # 加载神经网络模型
        self.init_scen_tree_gen()  # 初始化场景树生成器
        self.init_traj_tree_opt()  # 初始化轨迹树优化器

    def init_device(self):
        """根据配置选择计算设备 (CPU/GPU)"""
        if self.planner_cfg["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)  # 使用第一个GPU
        else:
            self.device = torch.device("cpu")  # 回退到CPU

    def init_network(self):
        """加载预训练神经网络模型"""
        # 动态导入网络配置
        self.network_cfg = import_module(self.planner_cfg["network_config"]).NetCfg()
        # 构建网络结构
        net_cfg = self.network_cfg.get_net_cfg()
        net_file, net_name = net_cfg["network"].split(":")  # 解析网络定义文件
        
        # 加载神经网络预测模型
        self.network = getattr(import_module(net_file), net_name)(net_cfg, self.device)
        # 加载预训练权重
        ckpt = torch.load(self.planner_cfg["ckpt_path"], map_location=lambda storage, loc: storage)
        self.network.load_state_dict(ckpt["state_dict"])
        self.network.eval()  # 设置为评估模式

    def init_scen_tree_gen(self):
        scen_tree_cfg = import_module(self.planner_cfg["planning_config"]).ScenTreeCfg()
        self.scen_tree_gen = ScenarioTreeGenerator(self.device, self.network, self.obs_len, self.plan_len, scen_tree_cfg)

    def init_traj_tree_opt(self):
        traj_tree_cfg = import_module(self.planner_cfg["planning_config"]).TrajTreeCfg()
        self.traj_tree_opt = TrajectoryTreeOptimizer(traj_tree_cfg)
        passss = 1

    def to_object_state(self, agent):
        obj_state = ObjectState(
            True,
            agent.timestep,
            (agent.state[0], agent.state[1]),
            agent.state[3],
            (agent.state[2] * np.cos(agent.state[3]), agent.state[2] * np.sin(agent.state[3])),
        )
        return obj_state

    def update_observation(self, lcl_smp):
        #  update ego agent
        if "AV" not in self.agent_obs:
            self.agent_obs["AV"] = Track("AV", [self.to_object_state(lcl_smp.ego_agent)], lcl_smp.ego_agent.type, TrackCategory.FOCAL_TRACK)
        else:
            self.agent_obs["AV"].object_states.append(self.to_object_state(lcl_smp.ego_agent))

        #  update exo agents
        updated_agent_ids = ["AV"]
        for agent in lcl_smp.exo_agents:
            if agent.id not in self.agent_obs:
                self.agent_obs[agent.id] = Track(agent.id, [self.to_object_state(agent)], agent.type, TrackCategory.TRACK_FRAGMENT)
            else:
                self.agent_obs[agent.id].object_states.append(self.to_object_state(agent))
            updated_agent_ids.append(agent.id)

        # assign dummy agents for agents that are not observed
        for agent in self.agent_obs.values():
            if agent.track_id not in updated_agent_ids:
                agent.object_states.append(
                    ObjectState(
                        False,
                        agent.object_states[-1].timestep,
                        agent.object_states[-1].position,
                        agent.object_states[-1].heading,
                        agent.object_states[-1].velocity,
                    )
                )

        for agent in self.agent_obs.values():
            if len(agent.object_states) > self.obs_len:
                agent.object_states.pop(0)

    def update_state_ctrl(self, state, ctrl):
        self.state = state
        self.ctrl = ctrl

    def update_target_lane(self, gt_tgt_lane):
        self.gt_tgt_lane = gt_tgt_lane

    def plan(self, lcl_smp):
        """规划主流程
        输入：
            lcl_smp: LocalSemanticMap对象，包含局部语义信息
        返回：
            (是否成功, 最优控制指令, [场景树,轨迹树])
        """
        t0 = time.time()
        # reset
        self.scen_tree_gen.reset()  # 重置场景树生成器

        # 目标车道重采样
        # high-level command: resampled target lane
        resample_target_lane, resample_target_lane_info = self.resample_target_lane(lcl_smp)
        self.scen_tree_gen.set_target_lane(resample_target_lane, resample_target_lane_info)

        # 1. 生成候选场景树 ========================================
        scen_trees = self.scen_tree_gen.branch_aime(lcl_smp, self.agent_obs)

        if len(scen_trees) < 0:
            return False, None, None

        #   2. 轨迹优化与评估（为每个场景树分支）========================
        traj_trees = []
        debug_info = []
        for scen_tree in scen_trees:
            traj_tree, debug = self.get_traj_tree(scen_tree, lcl_smp)
            traj_trees.append(traj_tree)
            debug_info.append(debug)

        # use multi-threading to speed up
        # n_proc = len(scen_trees)
        # traj_trees = Parallel(n_jobs=n_proc)(
        #     delayed(self.get_traj_tree)(scen_tree, lcl_smp) for scen_tree in scen_trees)

        # 3. 多目标决策选择最优轨迹 # select the best trajectory ========================
        best_traj_idx = None
        min_cost = np.inf
        for idx, traj_tree in enumerate(traj_trees):
            cost = self.evaluate_traj_tree(lcl_smp, traj_tree)
            if cost < min_cost:
                min_cost = cost
                best_traj_idx = idx

        # 4. 提取最优控制指令 ========================
        opt_traj_tree = traj_trees[best_traj_idx]
        next_node = opt_traj_tree.get_node(opt_traj_tree.get_root().children_keys[0])
        ret_ctrl = next_node.data[0][-2:]

        return True, ret_ctrl, [[scen_trees[best_traj_idx]], [traj_trees[best_traj_idx]]]

    def resample_target_lane(self, lcl_smp):
        """目标车道重采样
        输入：
            lcl_smp: LocalSemanticMap对象
        返回：
            (resample_target_lane, resample_target_lane_info) - 重采样后的车道坐标及属性
        """
        # resample the lcl_smp target_lane and info with 1.0m interval
        resample_target_lane = []
        resample_target_lane_info = [[] for _ in range(6)]

        # 对每个车道段进行线性插值
        for i in range(len(lcl_smp.target_lane) - 1):
            lane_segment = lcl_smp.target_lane[i : i + 2]
            lane_segment_len = np.linalg.norm(lane_segment[0] - lane_segment[1])
            num_sample = int(np.ceil(lane_segment_len / 1.0))  # 每米采样一个点
            for j in range(num_sample):
                alpha = j / num_sample
                resample_target_lane.append(lane_segment[0] + alpha * (lane_segment[1] - lane_segment[0]))
                # 同步车道属性信息
                for k, info in enumerate(lcl_smp.target_lane_info):
                    resample_target_lane_info[k].append(info[i])

        resample_target_lane.append(lcl_smp.target_lane[-1])
        for k, info in enumerate(lcl_smp.target_lane_info):
            resample_target_lane_info[k].append(info[-1])

        # to numpy
        resample_target_lane = np.array(resample_target_lane)
        for i in range(len(resample_target_lane_info)):
            resample_target_lane_info[i] = np.array(resample_target_lane_info[i])

        return resample_target_lane, resample_target_lane_info

    def get_traj_tree(self, scen_tree, lcl_smp):
        self.traj_tree_opt.init_warm_start_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity)
        xs, us = self.traj_tree_opt.warm_start_solve()
        self.traj_tree_opt.init_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity)
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
        comfort_acc_weight = 0.1
        comfort_str_weight = 5.0
        comfort_cost = 0.0
        efficiency_weight = 0.01
        efficiency_cost = 0.0
        target_weight = 0.01
        target_cost = 0.0

        n_nodes = len(traj_tree.nodes)
        for node in traj_tree.nodes.values():
            state = node.data[0]  # 状态向量 [x, y, v, θ, a, δ]
            ctrl = node.data[1]  # 状态向量 [x, y, v, θ, a, δ]

            # 舒适性计算 (加速度和转向变化惩罚)
            comfort_cost += comfort_acc_weight * ctrl[0] ** 2
            comfort_cost += comfort_str_weight * ctrl[1] ** 2

            # 效率计算 (速度偏差惩罚)
            efficiency_cost += efficiency_weight * (lcl_smp.target_velocity - state[2]) ** 2

            # 目标追踪计算 (车道中心线偏离惩罚)
            target_cost += target_weight * self.get_dist_to_target_lane(lcl_smp, state)
        return (comfort_cost + efficiency_cost + target_cost) / n_nodes

    def get_dist_to_target_lane(self, lcl_smp, state):
        #  project the state to the target lane
        proj_state, _, _ = project_point_on_polyline(state[:2], lcl_smp.target_lane)
        #  get the distance
        dist = np.linalg.norm(proj_state - state[:2])
        return dist

    def get_interpolated_state(self, tree, timestep):
        root_node = tree.get_node(0)
        if timestep < root_node.data.t:
            return root_node.data.state, root_node.data.ctrl
        else:
            node = root_node
            while node.data.t <= timestep:
                node = tree.get_node(node.children_keys[0])
            #  interpolate the state
            prev_node = tree.get_node(node.parent_key)
            prev_state = prev_node.data.state
            next_state = node.data.state
            prev_time = prev_node.data.t
            next_time = node.data.t
            alpha = (timestep - prev_time) / (next_time - prev_time)
            interp_state = prev_state + alpha * (next_state - prev_state)
            return interp_state, node.data.ctrl
