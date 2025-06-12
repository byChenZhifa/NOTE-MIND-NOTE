import copy
import numpy as np


# 3
from av2.datasets.motion_forecasting.data_schema import ObjectType

# local
from common.semantic_map import LocalSemanticMap
from planners.mind.planner import MINDPlanner
from common.bbox import PedestrianBBox, CyclistBBox, VehicleBBox, BusBBox, UnknownBBox
from common.kinematics import VehicleParam, kine_propagate
from common.geometry import project_point_on_polyline, remove_close_points


class AgentColor:
    """定义不同智能体的显示颜色方案"""

    def exo(self):
        """外部智能体（其他交通参与者）的颜色方案"""
        # 其他交通参与者的颜色（珊瑚红/深红）
        return ["lightcoral", "indianred"]  # facecolor, edgecolor

    def ego_disable(self):
        """未启用状态下的自车颜色方案"""
        # 不启用状态下的自车颜色（亮绿/蓝色）
        return ["lightskyblue", "deepskyblue"]

    def ego_enable(self):
        """启用状态下的自车颜色方案"""
        # 启用状态下的自车颜色（亮绿/蓝色）
        return ["lime", "blue"]  # 面颜色, 边颜色

    def interest(self):
        """感兴趣目标（如关键障碍物）的颜色方案"""
        return ["yellow", "orange"]

    def get_color(self, clr_name):
        """根据名称获取颜色方案"""
        if clr_name == "yellow":
            return ["yellow", "orange"]
        # 可以扩展其他颜色方案
        return


class AgentObservation:
    """封装单个交通参与者的观测数据结构"""

    # 封装单个交通参与者的观测数据
    def __init__(self):
        self.id = None  # 智能体唯一ID
        self.type = None  # 智能体类型（如车辆、行人等）
        self.clr = None  # 显示颜色方案
        self.bbox = None  # 包围盒对象
        self.state = None  # 状态向量 [x, y, 速度, 航向角]
        self.timestep = None  # 当前时间步长


class PlainAgent:
    """所有智能体的基类，实现基本观测功能"""

    def __init__(self):
        self.id = None
        self.type = None
        self.clr = None
        self.state = None
        self.ctrl = None
        self.bbox = None
        self.timestep = None

    def observe(self):
        """生成带噪声的观测（当前噪声被注释）"""
        obs = AgentObservation()  # 创建观测对象
        obs.id = self.id
        obs.type = self.type
        obs.clr = self.clr
        obs.state = self.state

        # 噪声添加（当前被注释掉）
        # noise = np.random.normal(0, 0.05, self.state.shape)
        # noise[-1] = 0.0  # 不对航向角添加噪声
        # obs.state = self.state + noise

        obs.bbox = self.bbox
        obs.timestep = self.timestep
        return obs

    def observe_no_noise(self):  # 生成原始观测
        """生成原始观测数据（无噪声）"""
        obs = AgentObservation()  # 创建观测对象
        obs.id = self.id
        obs.type = self.type
        obs.clr = self.clr
        obs.bbox = self.bbox
        obs.state = self.state
        obs.timestep = self.timestep

        return obs


class NonReactiveAgent(PlainAgent):
    """4. NonReactiveAgent（被动代理类）
    回放预定义轨迹的非反应式代理（如背景车辆）
    被动代理（背景车辆）回放预设轨迹
    """

    def __init__(self):
        super(NonReactiveAgent, self).__init__()  # 调用基类初始化
        self.traj_info = None  # 轨迹信息（位置，航向，速度等）
        self.traj_type = None  # 轨迹点对应的类型序列
        self.traj_cat = None  # 轨迹类别
        self.rec_step = 0  # 当前回放步数索引
        self.max_step = 0  # 最大轨迹步数
        self.lcl_smp = None  # 局部语义地图（被动智能体不使用）

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr):
        """使用轨迹数据初始化智能体

        参数:
            agt_id: 智能体唯一ID
            traj_type: 轨迹点对应的类型序列（ObjectType枚举）
            traj_cat: 轨迹类别
            traj_info: 轨迹信息元组 (位置数组, 航向数组, 速度数组)
            smp: 语义地图对象
            clr: 颜色方案
        """
        self.id = agt_id
        self.clr = clr
        self.traj_type = traj_type
        self.traj_cat = traj_cat
        self.traj_info = traj_info
        self.type = self.traj_type[self.rec_step]  # 当前类型

        # 根据类型创建相应的包围盒
        if self.type == ObjectType.VEHICLE:
            self.bbox = VehicleBBox()
        elif self.type == ObjectType.PEDESTRIAN:
            self.bbox = PedestrianBBox()
        elif self.type == ObjectType.MOTORCYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.CYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.BUS:
            self.bbox = BusBBox()
        elif self.type == ObjectType.UNKNOWN:
            self.bbox = UnknownBBox()
        else:  # 未知类型使用默认包围盒
            self.bbox = UnknownBBox()  # for all static objects

        # 从轨迹信息提取当前位置状态
        traj_pos, traj_ang, traj_vel = self.traj_info[:3]
        self.state = np.array(
            traj_pos[self.rec_step][0],  # x坐标
            traj_pos[self.rec_step][1],  # y坐标
            traj_vel[self.rec_step],  # 速度
            traj_ang[self.rec_step],  # 航向角
        )
        self.ctrl = np.array([0.0, 0.0])  # 被动智能体控制指令为零
        self.max_step = len(self.traj_info[0]) - 1  # 计算轨迹总步数
        self.lcl_smp = LocalSemanticMap(self.id, smp)  # 初始化局部语义地图（被动智能体不使用）
        self.timestep = 0.0  # 重置时间步
        # print("[Agent]: id: {} Initialized with traj_len:{}.".format(self.id, len(self.traj_info[0])))

    def check_trigger(self, sim_time):
        """检查是否触发回放推进（对被动智能体始终返回True）"""
        return True  # 被动智能体每步都推进

    def step(self):
        """推进轨迹索引到下一位置"""
        if self.rec_step >= self.max_step:  # 检查是否完成回放
            print("[Agent]: No.{} replay finished.".format(self.id))
            return
        self.rec_step += 1

    def update_state(self, dt):
        """更新状态到当前轨迹点"""
        # 从预录轨迹获取下一状态
        self.type = self.traj_type[self.rec_step]

        # 根据类型更新包围盒
        if self.type == ObjectType.VEHICLE:
            self.bbox = VehicleBBox()
        elif self.type == ObjectType.PEDESTRIAN:
            self.bbox = PedestrianBBox()
        elif self.type == ObjectType.MOTORCYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.CYCLIST:
            self.bbox = CyclistBBox()
        elif self.type == ObjectType.BUS:
            self.bbox = BusBBox()
        elif self.type == ObjectType.UNKNOWN:
            self.bbox = UnknownBBox()
        else:  # 未知类型使用默认包围盒
            self.bbox = UnknownBBox()  # for all static objects

        # 从轨迹信息更新状态
        traj_pos, traj_ang, traj_vel = self.traj_info[:3]
        self.state = np.array(
            [
                traj_pos[self.rec_step][0],  # x坐标
                traj_pos[self.rec_step][1],  # y坐标
                traj_vel[self.rec_step],  # 速度
                traj_ang[self.rec_step],  # 航向角
            ]
        )
        self.ctrl = np.array([0.0, 0.0])  # 控制指令清零
        self.timestep += dt  # 更新时间步

    def is_valid(self):
        """检查当前轨迹点是否有效"""
        return self.traj_info[-1][self.rec_step]  # 返回有效性标志


class CustomizedAgent(NonReactiveAgent):
    """5. CustomizedAgent（可定制代理类）
    可定制智能体基类（如自车），支持路径规划
    """

    def __init__(self):
        super(CustomizedAgent, self).__init__()  # 调用基类初始化
        self.last_pl_tri = None  # 上一次规划触发时间
        self.plan_rate = 10  # 规划频率 (Hz)
        self.plan_step = 1.0 / self.plan_rate - 1e-4  # 规划时间间隔 (秒)
        self.planner = None  # 规划器对象
        self.veh_param = VehicleParam()  # 车辆物理参数
        self.enable_timestep = 1e8  # 启用规划的时间点（默认不启用）
        self.is_enable = False  # 当前是否启用规划

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj=True, semantic_lane_id=None, target_velocity=None):
        """扩展初始化方法，融合轨迹数据和语义地图创建目标车道

        参数:
            agt_id: 智能体唯一ID
            traj_type: 轨迹点对应的类型序列
            traj_cat: 轨迹类别
            traj_info: 轨迹信息元组
            smp: 语义地图对象
            clr: 颜色方案
            use_traj: 是否使用轨迹生成目标车道
            semantic_lane_id: 指定语义车道ID
            target_velocity: 目标速度
        """
        super(CustomizedAgent, self).init(agt_id, traj_type, traj_cat, traj_info, smp, clr)

        # compute target lane by extending the recorded trajectory to the semantic lane
        # 融合历史轨迹与语义地图生成目标车道
        virtual_semantic_lane, virtual_semantic_lane_info = self.get_target_lane(smp, use_traj, semantic_lane_id)

        #  compute target velocity
        if target_velocity is None:
            target_velocity = np.mean(self.traj_info[2], axis=0)

        # 初始化局部语义地图
        self.lcl_smp = LocalSemanticMap(self.id, smp)
        self.lcl_smp.update_target_lane(virtual_semantic_lane)
        if virtual_semantic_lane_info is not None:
            self.lcl_smp.update_target_lane_info(virtual_semantic_lane_info)
        self.lcl_smp.update_target_velocity(target_velocity)
        self.timestep = 0.0
        self.init_state_ctrl()

    def get_target_lane(self, smp, use_traj, semantic_lane_id):
        """融合历史轨迹与语义地图生成目标车道

        参数：
            smp - 全局语义地图
            use_traj - 是否融合历史轨迹
            semantic_lane_id - 指定语义车道ID
        返回：
            virtual_target_lane - 虚拟目标车道坐标序列
            lane_info - 车道属性信息
        """
        traj_pos, traj_ang = self.traj_info[:2]  # 提取位置和航向数据

        # 当未指定车道ID时，自动匹配最近语义车道 ==========
        if semantic_lane_id is None:  # get the closest semantic lane
            # 通过轨迹起止点投影匹配语义车道
            semantic_lane_id = self.get_closest_semantic_lane(smp, traj_pos, traj_ang)
            if semantic_lane_id is None:  # use the historical trajectory as the target lane
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)
                # extending the historical trajectory as the semantic lane
                extend_pos = virtual_target_lane[-1] + (virtual_target_lane[-1] - virtual_target_lane[-2]) * 10.0
                virtual_target_lane = np.vstack([virtual_target_lane, extend_pos])
                return virtual_target_lane, None

            # 融合策略：将历史轨迹延伸部分与语义车道拼接 ==========
            if use_traj:
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)

                # 寻找轨迹终点在语义车道上的最近点
                # find the closest point on the semantic lane to the last pos of the historical trajectory
                closest_idx = np.argmin(np.linalg.norm(smp.semantic_lanes[semantic_lane_id] - traj_pos[-1], axis=1))

                # 拼接虚拟目标车道和语义车道
                virtual_target_lane = np.vstack([virtual_target_lane, smp.semantic_lanes[semantic_lane_id][closest_idx:]])

                return virtual_target_lane, None
            else:
                return smp.semantic_lanes[semantic_lane_id], smp.semantic_lanes_infos[semantic_lane_id]
        else:
            if semantic_lane_id not in smp.semantic_lanes:
                raise ValueError("Semantic lane id {} not in the semantic map.".format(semantic_lane_id))
            if use_traj:
                virtual_target_lane = self.get_virtual_target_lane(traj_pos)
                # 计算虚拟目标车道和语义车道的最近点
                # merge the virtual target lane with the semantic lane from the pos that is closest to the semantic lane
                diff = virtual_target_lane[:, np.newaxis, :] - smp.semantic_lanes[semantic_lane_id][np.newaxis, :, :]
                # compute the squared distance for each pair of points
                squared_distances = np.sum(diff**2, axis=2)
                # find the index of the minimum squared distance
                min_distance_index = np.argmin(squared_distances)  # 找到最小距离索引
                # convert the index into two-dimensional indices corresponding to the positions in virtual_target_lane and semantic lane
                vir_idx, sml_idx = np.unravel_index(min_distance_index, squared_distances.shape)  # 解包索引

                # 拼接虚拟目标车道和语义车道
                # truncate the virtual target lane to the closest point
                virtual_target_lane = virtual_target_lane[: vir_idx + 1]
                # merge the virtual target lane with the semantic lane from the closest point
                virtual_target_lane = np.vstack([virtual_target_lane, smp.semantic_lanes[semantic_lane_id][sml_idx:]])
                return virtual_target_lane, None
            else:
                return smp.semantic_lanes[semantic_lane_id], smp.semantic_lanes_infos[semantic_lane_id]

    def get_closest_semantic_lane(self, smp, traj_pos, traj_ang):
        """通过投影和角度约束寻找最合适的语义车道"""
        # compute target lane by extending the historical lane to the semantic lane
        closest_lane_id = None  # 初始化为None
        min_dis_diff = 1e9  # 最小距离差（初始设置为大数）
        ang_thres = np.pi / 4.0  # 角度阈值 (45度)
        dis_thres = 5.0  # 距离阈值 (米)

        # 遍历地图中的所有语义车道
        for lane_idx, lane in smp.semantic_lanes.items():
            # 计算轨迹起点在车道上的投影
            start_proj_pt, start_proj_heading, _ = project_point_on_polyline(traj_pos[0], lane)

            # 计算投影点角度与轨迹起始角度的差值
            start_ang_diff = np.abs(start_proj_heading - traj_ang[0])
            start_ang_diff = np.arctan2(np.sin(start_ang_diff), np.cos(start_ang_diff))

            # 计算投影点与轨迹起点的距离
            start_dis_diff = np.linalg.norm(traj_pos[0] - start_proj_pt)
            # 筛选项角度或距离超界的车道
            if start_dis_diff > dis_thres or start_ang_diff > ang_thres:
                continue

            # 计算轨迹终点在车道上的投影
            end_proj_pt, end_proj_heading, _ = project_point_on_polyline(traj_pos[-1], lane)
            # 计算角度差并标准化
            # cal angle diff with normalization to [-pi, pi]
            end_ang_diff = np.abs(end_proj_heading - traj_ang[-1])
            end_ang_diff = np.arctan2(np.sin(end_ang_diff), np.cos(end_ang_diff))

            # 计算终点距离
            end_dis_diff = np.linalg.norm(traj_pos[-1] - end_proj_pt)
            # 如果满足终点条件且是最近车道
            if end_ang_diff < ang_thres and end_dis_diff < dis_thres:
                if end_dis_diff < min_dis_diff:  # 更新最小距离
                    min_dis_diff = end_dis_diff
                    closest_lane_id = lane_idx
        return closest_lane_id  # 返回最接近的车道ID

    def get_virtual_target_lane(self, traj_pos):
        """通过简化历史轨迹创建虚拟目标车道"""
        # compute target lane by extending the historical lane to the semantic lane
        simplify_thres = 0.1
        traj_pos = remove_close_points(traj_pos, simplify_thres)  # 移除过于接近的点
        virtual_semantic_lane = copy.deepcopy(traj_pos)  # 深度拷贝避免修改原数据
        return virtual_semantic_lane

    def set_enable_timestep(self, timestep):
        """设置启用规划的时间点"""
        self.enable_timestep = timestep

    def check_enable(self, timestep):
        """检查是否应该启用规划功能"""
        if timestep >= self.enable_timestep and not self.is_enable:
            self.is_enable = True  # 启用规划
            self.init_state_ctrl()  # 重置状态和控制
            # 启用时可更换颜色方案（可选）
            # self.clr = AgentColor().ego_enable()  # change the color to enable color

    def init_state_ctrl(self):
        """从轨迹信息初始化状态和控制"""
        #  get initial state from the cfg
        traj_pos, traj_ang, traj_vel = self.traj_info[:3]  # 提取轨迹数据
        # 设置初始状态
        self.state = np.array(
            [
                traj_pos[self.rec_step][0],  # x坐标
                traj_pos[self.rec_step][1],  # y坐标
                traj_vel[self.rec_step],  # 速度
                traj_ang[self.rec_step],  # 航向角
            ]
        )
        self.ctrl = np.array([0.0, 0.0])  # 初始控制设为0

    def init_planner(self, cfg_dir):
        pass

    def check_trigger(self, sim_time):
        """控制记录推进和规划触发时机"""
        record_trigger = False  # 记录推进触发标志
        planner_trigger = False  # 规划触发标志

        # 如果未启用规划，则使用基类的记录触发逻辑
        if not self.is_enable:
            record_trigger = super().check_trigger(sim_time)

        #!检查是否到达规划时间间隔
        # 0.1s触发一次规划 sim_time=0.74  self.last_pl_tri=0.7 self.plan_step=0.1
        if self.last_pl_tri is None or (sim_time - self.last_pl_tri) >= self.plan_step:
            planner_trigger = True
            self.last_pl_tri = sim_time  # 更新最后规划时间

        return record_trigger, planner_trigger

    def plan(self):
        """规划方法（需子类实现）"""
        return True, None  # 基类返回成功但无实际规划

    def update_state(self, dt):
        """根据是否启用规划选择不同的状态更新方法"""
        if not self.is_enable:  # 未启用规划时使用轨迹回放
            super().update_state(dt)
        else:  # 启用规划时使用动力学模型
            self._update_state(dt)

    def _update_state(self, dt):
        """使用车辆运动学模型更新状态
                基于车辆运动学模型更新状态
        使用公式：x' = x + v*cosθ*dt
                 y' = y + v*sinθ*dt
                 θ' = θ + (v/L)*tan(steer)*dt
        参数：
            dt - 时间步长
            veh_param - 车辆参数（轴距、最大速度等）
        """
        self.state = kine_propagate(
            self.state,
            self.ctrl,  # 控制量 [加速度, 转向角]
            dt,
            self.veh_param.wb,  # 轴距
            self.veh_param.max_spd,
            self.veh_param.max_str,  # 最大转向角
        )
        self.timestep += dt

    def update_observation(self, agents):
        """更新局部语义地图中的观测信息"""
        self.lcl_smp.update_observation(agents)


class MINDAgent(CustomizedAgent):
    """使用MIND规划算法的具体实现（自车）"""

    def __init__(self):
        super(MINDAgent, self).__init__()  # 调用基类初始化
        self.gt_tgt_lane = None  # 真实目标车道（用于规划）

    def init(self, agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj=False, semantic_lane_id=None, target_velocity=None):
        """仅使用语义地图作为目标车道"""
        # 调用基类初始化（仅使用语义地图）
        super().init(agt_id, traj_type, traj_cat, traj_info, smp, clr, use_traj, semantic_lane_id, target_velocity)

    def init_planner(self, cfg_dir):
        """初始化MIND规划器"""
        self.planner = MINDPlanner(cfg_dir)  # 创建MIND规划器对象

    def update_target_lane(self, smp, semantic_lane_id):
        """更新真实目标车道用于规划"""
        self.gt_tgt_lane, _ = self.get_target_lane(smp, True, semantic_lane_id)  # 获取目标车道
        self.gt_tgt_lane = remove_close_points(self.gt_tgt_lane, 4.0)  # 简化车道点
        self.planner.update_target_lane(self.gt_tgt_lane)  # 通知规划器更新目标车道

    def plan(self):
        """调用MIND规划器生成控制指令"""
        # 更新规划器的状态和控制
        self.planner.update_state_ctrl(state=self.lcl_smp.ego_agent.state, ctrl=self.ctrl)
        # 执行规划
        is_success, self.ctrl, best_tree_set = self.planner.plan(self.lcl_smp)
        return is_success, best_tree_set

    def update_observation(self, agents):
        """更新规划器中的观测信息"""
        # 先更新局部语义地图
        super().update_observation(agents)
        # 再通知规划器更新观测
        self.planner.update_observation(self.lcl_smp)
        