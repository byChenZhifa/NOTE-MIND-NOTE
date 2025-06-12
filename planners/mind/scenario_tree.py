import copy
import torch
import numpy as np
from planners.basic.tree import Tree, Node
from planners.mind.utils import (
    gpu,
    from_numpy,
    get_max_covariance,
    get_origin_rotation,
    get_new_lane_graph,
    get_rpe,
    get_angle,
    collate_fn,
    get_agent_trajectories,
    update_lane_graph_from_argo,
    get_distance_to_polyline,
)


class ScenarioData:
    """场景数据容器，存储预测结果及元数据"""

    def __init__(self, data, obs_data, branch_flag=False, end_flag=False, terminate_flag=False):
        """
        参数:
            data: 预测数据
            obs_data: 观测数据
            branch_flag: 是否可分支标志
            end_flag: 是否终止标志
            terminate_flag: 是否提前终止标志
        """
        self.data = data  # 主要数据（预测结果）
        self.obs_data = obs_data  # 观测数据（用于后续预测）
        self.branch_flag = branch_flag  # 该节点是否可进行分支
        self.end_flag = end_flag  # 该节点是否为分支终点
        self.terminate_flag = terminate_flag  # 该节点是否提前终止


class ScenarioTreeGenerator:
    """场景树生成器，负责创建交互预测树"""

    def __init__(self, device, network, obs_len=50, pred_len=60, config=None):
        """
        参数:
            device: 计算设备 (CPU/GPU)
            network: 预测网络模型
            obs_len: 历史观测长度（时间步）
            pred_len: 预测长度（时间步）
            config: 配置参数
        """
        self.device = device  # 计算设备
        self.network = network  # 轨迹预测神经网络
        self.obs_len = obs_len  # 观测历史长度
        self.pred_len = pred_len  # 预测长度
        self.seq_len = self.obs_len + self.pred_len  # 总序列长度
        self.config = config  # 配置对象
        self.tree = Tree()  # 场景树数据结构
        self.lane_graph = None  # 车道图数据
        self.target_lane = None  # 目标车道点
        self.target_lane_info = None  # 目标车道信息
        self.ego_idx = 0  # 自车索引位置
        self.branch_depth = 0  # 当前分支深度

    def reset(self):
        """重置场景树生成器状态"""
        self.branch_depth = 0  # 重置分支深度
        self.tree = Tree()  # 创建新树

    def branch_aime(self, lcl_smp, agent_obs):
        """AIME算法主流程
        输入：
            lcl_smp: 局部语义地图
            agent_obs: 所有交通参与者的观测数据

        返回：
            生成的场景树集合
        """

        # 1. 数据预处理 Initialization
        data = self.process_data(lcl_smp, agent_obs)
        #  2. 初始化场景树
        self.init_scenario_tree(data)

        # 3. 迭代式分支生成
        # AIME iteration
        branch_nodes = self.get_branch_set()
        while branch_nodes:
            # 3.1. 神经网络预测多智能体未来轨迹 ====================
            # Batch Scenario Prediction  批量预测场景发展
            data_batch = collate_fn([node.data.obs_data for node in branch_nodes])
            pred_batch = self.predict_scenes(data_batch)

            # 3.2. 场景剪枝与合并 Pruning & Merging  剪枝与合并  ====================
            pred_bar = self.prune_merge(data_batch, pred_batch)

            # 创建新节点
            # Create New Nodes (slightly different from the pseudocode in paper)
            self.create_nodes(pred_bar)

            # 3.3. 创建场景树节点  ====================
            # Branching Decision on newly added node  创建新节点; 更新分支决策
            self.decide_branch()
            # Update Branch Set  更新可分支节点集合
            branch_nodes = self.get_branch_set()

        # 确保有终止节点
        assert len(self.get_end_set()) > 0, "No end node found in the scenario tree."

        # 返回生成的场景树
        return self.get_scenario_tree()

    def init_scenario_tree(self, data):
        """使用初始数据初始化场景树"""
        # 准备根节点数据
        # prepossess the observation data and map data
        root_data = self.prepare_root_data(data)

        # 添加根节点到场景树
        self.tree.add_node(Node("root", None, ScenarioData(None, root_data, branch_flag=True)))

        # 进行初步预测
        pred_batch = self.predict_scenes(root_data)

        # 剪枝与合并初步预测
        pred_bar = self.prune_merge(root_data, pred_batch)

        # 创建初始节点
        self.create_nodes(pred_bar)

        # 初始分支决策
        self.decide_branch()

    def predict_scenes(self, data):
        """使用神经网络预测未来场景"""
        data_in = self.network.pre_process(data)  # 数据预处理
        return self.network(data_in)  # 神经网络前向传播

    def create_nodes(self, pred_bar):
        """根据预测结果创建新的场景树节点"""
        for pred in pred_bar:  # 遍历所有预测结果
            parent_id = pred["PARENT_ID"]  # 父节点ID
            node_id = pred["SCEN_ID"]  # 新节点ID

            # 创建新节点并添加到树  Create new node
            new_node = Node(node_id, parent_id, ScenarioData(pred, None))
            # Attach to the tree
            self.tree.add_node(new_node)

    def decide_branch(self):
        """决定哪些节点需要进行分支"""
        # iterate over the leaf nodes  遍历所有叶节点
        for leaf_node in self.tree.get_leaf_nodes():
            if leaf_node.data.branch_flag:  # 如果当前节点标记为可分支
                leaf_node.data.branch_flag = False  # 重置分支标志
                leaf_node.data.terminate_flag = True  # 标记为已终止
            elif not leaf_node.data.end_flag:  # 如果未到终止点
                if leaf_node.depth >= self.config.max_depth:  # 超过最大深度
                    leaf_node.data.terminate_flag = True  # 标记终止
                else:
                    # 计算分支时间点
                    t_b = self.get_branch_time(leaf_node.data.data)
                    if t_b < self.pred_len:  # 如果分支时间在预测长度内
                        # Update the observation data  更新观测数据
                        leaf_node.data.obs_data, leaf_node.data.data = self.update_obser(leaf_node.data.data)
                        # Add node to branch set  标记为可分支
                        leaf_node.data.branch_flag = True
                    else:  # 否则标记为终止节点
                        # Add node the end set
                        leaf_node.data.end_flag = True

    def get_branch_set(self):
        """获取所有可分支的节点集合"""
        branch_set = []  # 分支节点集合
        for node in self.tree.get_leaf_nodes():  # 遍历叶节点
            if node.data.branch_flag:  # 如果节点标记为可分支
                branch_set.append(node)  # 添加到集合

        # 增加分支深度计数
        self.branch_depth += 1
        return branch_set

    def set_target_lane(self, target_lane, target_lane_info):
        """设置目标车道信息"""
        # 转移到GPU并转换为tensor
        self.target_lane = gpu(torch.from_numpy(np.array(target_lane)), self.device)

        # 处理车道信息
        self.target_lane_info = torch.cat(
            [
                torch.from_numpy(target_lane_info[0]).unsqueeze(1),  # 类型信息
                torch.from_numpy(target_lane_info[1]),  # 几何信息
                torch.from_numpy(target_lane_info[2]),
                torch.from_numpy(target_lane_info[3]),
                torch.from_numpy(target_lane_info[4]).unsqueeze(1),  # 拓扑信息
                torch.from_numpy(target_lane_info[5]).unsqueeze(1),  # 其他属性
            ],
            dim=-1,
        )  # 沿最后维度拼接    # [N_{lane}, 16, F]

        # 转移到GPU
        self.target_lane_info = gpu(self.target_lane_info, self.device)

    def process_data(self, lcl_smp, agent_obs):
        """
        处理输入数据，准备神经网络预测格式

        返回:
            处理后的数据集合
        """
        # 获取智能体轨迹数据
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags, trajs_tid, trajs_cat = get_agent_trajectories(agent_obs, self.device)

        # 获取自车当前速度
        cur_vel = lcl_smp.ego_agent.state[2]

        # 计算坐标系变换（转换为以目标点为中心）
        orig_seq, rot_seq, theta_seq = get_origin_rotation(trajs_pos[0], trajs_ang[0], self.device)  # * target-centric

        # 获取并更新车道图
        # ~ get lane graph
        lane_graph = update_lane_graph_from_argo(lcl_smp.map_data, orig_seq.cpu().numpy(), rot_seq.cpu().numpy())

        lane_graph = gpu(from_numpy(lane_graph), self.device)

        # ====== 坐标系归一化 ======
        # 位置归一化（减去中心点）
        # ~ normalize w.r.t. scene
        trajs_pos = torch.matmul(trajs_pos - orig_seq, rot_seq)
        # 角度归一化（减去参考角度）
        trajs_ang = trajs_ang - theta_seq
        # 速度归一化（旋转速度向量）
        trajs_vel = torch.matmul(trajs_vel, rot_seq)

        # ====== 智能体坐标系归一化 ======
        # ~ normalize trajs
        trajs_pos_norm = []
        trajs_ang_norm = []
        trajs_vel_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj_pos, traj_ang, traj_vel in zip(trajs_pos, trajs_ang, trajs_vel):
            # 计算每个智能体的局部坐标系变换
            orig_act, rot_act, theta_act = get_origin_rotation(traj_pos, traj_ang, self.device)

            # 应用坐标变换
            trajs_pos_norm.append(torch.matmul(traj_pos - orig_act, rot_act))
            trajs_ang_norm.append(traj_ang - theta_act)
            trajs_vel_norm.append(torch.matmul(traj_vel, rot_act))
            trajs_ctrs.append(orig_act)  # 智能体坐标原点
            trajs_vecs.append(torch.tensor([torch.cos(theta_act), torch.sin(theta_act)]))  # 智能体方向向量

        # 重构为张量  应用坐标变换
        trajs_pos = torch.stack(trajs_pos_norm)  # 归一化位置 [N, 110(50), 2]
        trajs_ang = torch.stack(trajs_ang_norm)  # 归一化角度  [N, 110(50)]
        trajs_vel = torch.stack(trajs_vel_norm)  # 归一化速度 [N, 110(50), 2]
        trajs_ctrs = torch.stack(trajs_ctrs).to(self.device)  # 坐标原点 [N, 2]
        trajs_vecs = torch.stack(trajs_vecs).to(self.device)  # [N, 2] #智能体方向向量

        # 构建轨迹数据字典
        trajs = dict()
        # 观测数据部分
        trajs["TRAJS_POS_OBS"] = trajs_pos  # 位置观测
        trajs["TRAJS_ANG_OBS"] = torch.stack([torch.cos(trajs_ang), torch.sin(trajs_ang)], dim=-1)  # 角度转为向量
        trajs["TRAJS_VEL_OBS"] = trajs_vel  # 速度观测
        trajs["TRAJS_TYPE"] = trajs_type  # 智能体类型
        trajs["PAD_OBS"] = has_flags  # 数据有效标志
        # 坐标系转换相关   anchor ctrs & vecs
        trajs["TRAJS_CTRS"] = trajs_ctrs  # 位置坐标原点
        trajs["TRAJS_VECS"] = trajs_vecs  # 方向向量
        # 智能体元数据  track id & category
        trajs["TRAJS_TID"] = trajs_tid  # 轨迹ID列表
        trajs["TRAJS_CAT"] = trajs_cat  # 轨迹类别列表

        # 获取高级命令（目标车道点）
        tgt_pts, tgt_nodes, tgt_anch = self.get_high_level_command(orig_seq, rot_seq, cur_vel)

        # ====== 计算相对位置编码 ======
        lane_ctrs = lane_graph["lane_ctrs"]  # 车道中心点
        lane_vecs = lane_graph["lane_vecs"]  # 车道方向向量

        # 场景级RPE
        rpes = dict()
        scene_ctrs = torch.cat([trajs_ctrs, lane_ctrs], dim=0)  # 组合智能体和车道中心点
        scene_vecs = torch.cat([trajs_vecs, lane_vecs], dim=0)  # 组合智能体和车道方向
        rpes["scene"], rpes["scene_mask"] = get_rpe(scene_ctrs, scene_vecs)  # 计算场景RPE

        # 目标级RPE  ~ calc rpe for tgt
        tgt_ctr, tgt_vec = tgt_anch
        tgt_ctrs = torch.cat([tgt_ctr.unsqueeze(0), trajs_ctrs[0].unsqueeze(0)])  # 目标点和自车原点
        tgt_vecs = torch.cat([tgt_vec.unsqueeze(0), trajs_vecs[0].unsqueeze(0)])  # 目标方向和自车方向
        tgt_rpe, _ = get_rpe(tgt_ctrs, tgt_vecs)  # 计算目标RPE

        # 构建完整数据字典 prepare data
        data = {}
        data["ORIG"] = orig_seq  # 原点坐标
        data["ROT"] = rot_seq  # 旋转矩阵
        data["TRAJS"] = trajs  # 轨迹数据
        data["LANE_GRAPH"] = lane_graph  # 车道图
        data["TGT_PTS"] = tgt_pts  # 目标点
        data["TGT_NODES"] = tgt_nodes  # 目标节点
        data["TGT_ANCH"] = tgt_anch  # 目标锚点
        data["RPE"] = rpes  # 相对位置编码
        data["TGT_NODES"] = tgt_nodes  # 目标节点（二次存储）
        data["TGT_ANCH"] = tgt_anch  # 目标锚点（二次存储）
        data["TGT_RPE"] = tgt_rpe  # 目标相对位置编码

        # 保存车道图副本
        self.lane_graph = copy.deepcopy(data["LANE_GRAPH"])
        return gpu(collate_fn([data]), self.device)  # 返回批处理并转为GPU数据

    def get_scenario_tree(self):
        """构建最终场景树结构并计算概率分布"""
        data_tree = Tree()  # 创建新数据结构树
        root_node = self.tree.get_root()  # 获取场景树根节点

        # 添加根节点到数据结构树
        data_tree.add_node(Node(root_node.key, None, [1.0]))  # 根节点概率为1

        # label the branch that actually finished 标记实际终止的分支
        for node in self.get_end_set():  # 遍历所有终止节点
            while node.parent_key is not None:  # 向上遍历至根节点
                node.data.end_flag = True  # 标记为终止
                node = self.tree.get_node(node.parent_key)  # 移动到父节点

        # ====== 构建完整树结构 ======
        # 处理根节点子节点 construct the data_tree recursively and add normalized probability
        for key in root_node.children_keys:
            node = self.tree.get_node(key)  # 获取子节点
            if not node.data.end_flag:  # 跳过非终止节点
                continue

            # 添加节点到数据结构树
            data_tree.add_node(Node(node.key, root_node.key, [1.0]))  # 初始概率为1

            # BFS遍历构建树结构
            queue = [node]  # 初始化队列
            while queue:
                cur_node = queue.pop(0)  # 取出当前节点
                parent_prob = data_tree.get_node(cur_node.key).data[0]  # 获取父节点概率
                total_prob = 0.0  # 子节点总概率

                # 计算所有子节点概率和
                for child_key in cur_node.children_keys:
                    child_node = self.tree.get_node(child_key)
                    if child_node.data.end_flag:
                        total_prob += child_node.data.data["SCEN_PROB"].cpu().numpy()  # 累加概率

                # 处理每个子节点
                for child_key in cur_node.children_keys:
                    child_node = self.tree.get_node(child_key)
                    if child_node.data.end_flag:
                        # 计算归一化概率：子节点概率/总概率 * 父节点概率
                        # data_tree.add_node(
                        #     Node(child_node.key, cur_node.key, [child_node.data.data["SCEN_PROB"].cpu().numpy() / total_prob * parent_prob])
                        # )
                        child_prob = child_node.data.data["SCEN_PROB"].cpu().numpy() / total_prob * parent_prob

                        # 添加子节点到数据结构树
                        data_tree.add_node(Node(child_node.key, cur_node.key, [child_prob]))

                        # 添加到处理队列
                        queue.append(child_node)

        # ====== 添加轨迹数据到节点 ======
        # add traj, cov, tgt_lane to the data_tree
        for node in self.get_end_set():  # 遍历终止节点
            current_node = node
            while current_node.parent_key is not None:  # 向上遍历至根节点
                # 获取场景持续时间
                duration = current_node.data.data["END_T"] - current_node.data.data["CUR_T"]

                # 获取数据结构树中对应节点
                data_node = data_tree.get_node(current_node.key)

                # 如果节点只有概率数据，添加轨迹信息
                if len(data_node.data) == 1:
                    # 添加轨迹数据三元组：轨迹位置、轨迹协方差、目标点
                    data_node.data += [
                        current_node.data.data["TRAJS_POS_HIST"][:, self.obs_len : self.obs_len + duration, :].cpu().numpy(),
                        current_node.data.data["TRAJS_COV_HIST"][:, self.obs_len : self.obs_len + duration, :].cpu().numpy(),
                        current_node.data.data["TGT_PTS"].cpu().numpy(),
                    ]

                # 移动到父节点
                current_node = self.tree.get_node(current_node.parent_key)

        # ====== 分离完整场景树 ======
        #  separate the data_tree into trajectory trees from the root
        scenario_trees = []  # 存储最终场景树

        # 遍历根节点的所有子节点
        for key in data_tree.get_root().children_keys:
            scenario_tree = Tree()  # 新建树
            node = data_tree.get_node(key)  # 获取当前节点

            # 添加到新树
            scenario_tree.add_node(Node(node.key, None, node.data))

            # BFS遍历构建完整树
            #  add the children nodes recursively and add normalized probability
            queue = [node]  # 初始化队列
            while queue:
                cur_node = queue.pop(0)  # 取出现节点
                # 处理所有子节点
                for child_key in cur_node.children_keys:
                    child_node = data_tree.get_node(child_key)
                    # 添加到树
                    scenario_tree.add_node(Node(child_node.key, cur_node.key, child_node.data))
                    # 添加到队列
                    queue.append(child_node)

            # 添加到结果列表
            scenario_trees.append(scenario_tree)

        return scenario_trees  # 返回生成的场景树集合

    def get_end_set(self):
        """获取所有终止节点集合"""
        end_nodes = []  # 终止节点列表
        for node in self.tree.get_leaf_nodes():  # 遍历叶节点
            if node.data.end_flag:  # 如果是终止节点
                end_nodes.append(node)  # 添加到结果
        return end_nodes

    def prune_merge(self, data, out):
        """预测结果剪枝与相似场景合并"""
        data_interact = []  # 存储处理后的预测结果
        batch_size = len(data["ORIG"])  # 批量大小
        # 解包网络输出：分类结果（场景概率）、回归结果（位置）、辅助输出（速度/角度）
        res_cls_batch, res_reg_batch, res_aux_batch = out

        # 处理批量中的每个预测样本
        for idx in range(batch_size):
            # 提取数据元素
            orig = data["ORIG"][idx]  # 原点
            rot = data["ROT"][idx]  # 旋转矩阵
            trajs_ctrs = data["TRAJS"][idx]["TRAJS_CTRS"]  # 位置原点
            trajs_vecs = data["TRAJS"][idx]["TRAJS_VECS"]  # 方向向量
            trajs_type = data["TRAJS"][idx]["TRAJS_TYPE"]  # 智能体类型
            trajs_tid = data["TRAJS"][idx]["TRAJS_TID"]  # 轨迹ID
            trajs_cat = data["TRAJS"][idx]["TRAJS_CAT"]  # 轨迹类别

            # 计算全局坐标系角度
            theta_global = torch.atan2(rot[1, 0], rot[0, 0])

            # 提取历史轨迹数据
            trajs_pos_hist = data["TRAJS_POS_HIST"][idx]  # 位置历史
            trajs_ang_hist = data["TRAJS_ANG_HIST"][idx]  # 角度历史
            trajs_vel_hist = data["TRAJS_VEL_HIST"][idx]  # 速度历史
            trajs_cov_hist = data["TRAJS_COV_HIST"][idx]  # 协方差历史

            # 提取场景树元数据
            parent_id = data["PARENT_ID"][idx]  # 父节点ID
            parent_prob = data["SCEN_PROB"][idx]  # 父节点概率
            cur_t = data["CUR_T"][idx]  # 当前时间点
            end_t = data["END_T"][idx]  # 结束时间点

            # 处理预测结果
            res_reg = res_reg_batch[idx].detach()  # 位置预测
            res_cls = res_cls_batch[idx].detach()  # 场景概率预测
            res_vel = res_aux_batch[idx][0].detach()  # 速度预测
            res_ang = get_angle(res_vel)  # 角度计算

            # 按概率降序排列场景
            # sort the scene by the probability
            scene_idcs = torch.argsort(res_cls, dim=1, descending=True)[0]  # 获取排序索引

            data_candidates = []  # 候选场景列表

            # 处理每个预测场景
            for scene_id in scene_idcs:
                scene_prob = res_cls[0, scene_id]  # 场景概率
                scen_id = f"{self.branch_depth}_{idx}_{scene_id}"  # 生成场景ID

                # 提取预测位置和协方差
                trajs_pos_pred = res_reg[:, scene_id, :, :2]  # 预测位置
                # trajs_cov_pred = get_covariance_matrix(res_reg[:, scene_id, :, 2:])
                trajs_cov_pred = get_max_covariance(res_reg[:, scene_id, :, 2:])  # 最大协方差 use the max sigma
                trajs_vel_pred = res_vel[:, scene_id]  # 预测速度

                # 转换到全局坐标系
                trajs_theta = torch.atan2(trajs_vecs[:, 1], trajs_vecs[:, 0])  # 各智能体角度
                trajs_rots = (
                    torch.stack(
                        [torch.cos(trajs_theta), -torch.sin(trajs_theta), torch.sin(trajs_theta), torch.cos(trajs_theta)], dim=1  # 构建旋转矩阵
                    )
                    .view(-1, 2, 2)
                    .to(self.device)
                )

                # 应用智能体局部坐标系变换
                for i in range(len(trajs_pos_pred)):
                    # 位置变换
                    trajs_pos_pred[i] = torch.matmul(trajs_pos_pred[i], trajs_rots[i].transpose(-1, -2)) + trajs_ctrs[i]
                    # 速度变换
                    trajs_vel_pred[i] = torch.matmul(trajs_vel_pred[i], trajs_rots[i].transpose(-1, -2))
                    # 协方差变换（此处注释掉）
                    # trajs_cov_pred[i] = torch.matmul(trajs_rots[i],
                    #                                  torch.matmul(trajs_cov_pred[i], trajs_rots[i].transpose(-1, -2)))

                # 全局坐标变换
                trajs_pos_pred = torch.matmul(trajs_pos_pred, rot.T) + orig
                trajs_vel_pred = torch.matmul(trajs_vel_pred, rot.T)
                # 角度转换：局部角度 + 智能体方向 + 全局角度
                trajs_ang_pred = res_ang[:, scene_id] + trajs_theta.unsqueeze(1) + theta_global

                # 累积协方差（历史+预测）
                trajs_cov_pred += trajs_cov_hist[:, -1].unsqueeze(1)

                # 构建完整历史+预测轨迹
                trajs_pos_hist_new = torch.cat([trajs_pos_hist, trajs_pos_pred], dim=1)[:, : self.seq_len]
                trajs_cov_hist_new = torch.cat([trajs_cov_hist, trajs_cov_pred], dim=1)[:, : self.seq_len]
                trajs_ang_hist_new = torch.cat([trajs_ang_hist, trajs_ang_pred], dim=1)[:, : self.seq_len]
                trajs_vel_hist_new = torch.cat([trajs_vel_hist, trajs_vel_pred], dim=1)[:, : self.seq_len]

                # 构建轨迹数据
                cur_traj_data = dict()
                cur_traj_data["TRAJS_TYPE"] = trajs_type  # 智能体类型
                cur_traj_data["TRAJS_TID"] = trajs_tid  # 轨迹ID
                cur_traj_data["TRAJS_CAT"] = trajs_cat  # 轨迹类别

                # 构建完整场景数据
                cur_data = {}
                cur_data["SCEN_PROB"] = scene_prob * parent_prob  # 场景概率（乘父节点概率）
                cur_data["CUR_T"] = cur_t  # 当前时间点
                cur_data["END_T"] = end_t  # 结束时间点
                cur_data["PARENT_ID"] = parent_id  # 父节点ID
                cur_data["SCEN_ID"] = scen_id  # 场景ID
                cur_data["TRAJS"] = cur_traj_data  # 轨迹数据
                cur_data["TRAJS_POS_HIST"] = trajs_pos_hist_new  # 位置历史+预测
                cur_data["TRAJS_COV_HIST"] = trajs_cov_hist_new  # 协方差历史+预测
                cur_data["TRAJS_ANG_HIST"] = trajs_ang_hist_new  # 角度历史+预测
                cur_data["TRAJS_VEL_HIST"] = trajs_vel_hist_new  # 速度历史+预测
                cur_data["TGT_PTS"] = data["TGT_PTS"][idx]  # 目标点

                # ====== 剪枝条件 ======
                # 1. 概率过滤：跳过低概率场景
                # prune if the scene is not likely
                if cur_data["SCEN_PROB"] < 0.001:
                    continue

                # 2. 目标跟随过滤：检查自车目标跟随
                # prune if the ego decision is not likely to follow the target lane
                if self.target_lane is not None and self.ego_idx is not None:
                    # 获取自车位置和协方差
                    ego_mean = cur_data["TRAJS_POS_HIST"][self.ego_idx][-1]  # 自车平均位置
                    ego_cov = cur_data["TRAJS_COV_HIST"][self.ego_idx][-1]  # 自车协方差

                    # 计算到目标车道的距离
                    dis = get_distance_to_polyline(self.target_lane, ego_mean)

                    # 如果距离超过阈值（考虑协方差），跳过该场景
                    if dis - ego_cov > self.config.tar_dist_thres:
                        continue

                # ====== 合并条件 ======
                # 计算轨迹拓扑变化量
                # cal the topo cum change for merging
                topos = torch.zeros(len(trajs_pos_pred) - 1)  # 拓扑变化向量
                for exo_idx, traj in enumerate(trajs_pos_pred[1:]):  # 遍历非自车
                    # cal the cum angle change of the vector pointing from ego to the exo
                    # 计算从自车到非自车的方向向量
                    vec = traj - trajs_pos_pred[0]  # 相对位置向量
                    vec = vec / torch.norm(vec, dim=-1, keepdim=True)  # 归一化
                    ang = torch.atan2(vec[:, 1], vec[:, 0])  # 计算方向角度
                    ang_diff = ang[1:] - ang[:-1]  # 角度变化量
                    # normalize the angle diff
                    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))  # 标准化角度变化
                    # cal the cum angle change of the vector pointing from ego to the exo
                    topos[exo_idx] = torch.sum(ang_diff)  # 累积角度变化

                # 添加到候选列表
                data_candidates.append([cur_data, scene_prob, topos])

            # ====== 合并相似场景 ======
            selected_data = []  # 筛选后的场景
            min_topo_change = torch.pi / 6  # 最小拓扑变化阈值 (30度)
            while data_candidates:
                select_data, select_prob, select_topos = data_candidates[0]  # 取出第一个场景
                selected_data.append(select_data)  # 添加到最终结果
                data_candidates_tmp = []  # 新候选列表

                # 遍历剩余候选场景
                for candidate in data_candidates[1:]:
                    _, _, res_topos = candidate
                    # 计算拓扑变化差异
                    topos_diff = select_topos - res_topos
                    topos_diff = torch.atan2(torch.sin(topos_diff), torch.cos(topos_diff))  # 标准化
                    # 如果拓扑变化超过阈值，保留场景
                    if torch.any(torch.abs(topos_diff) > min_topo_change):
                        data_candidates_tmp.append(candidate)

                # 更新候选列表
                data_candidates = data_candidates_tmp

            # 添加筛选后的场景到最终结果
            data_interact += selected_data

        return data_interact  # 返回处理后场景

    def prepare_root_data(self, data):
        """准备根节点数据"""
        batch_size = len(data["ORIG"])  # 批量大小
        # 初始化轨迹历史数据结构
        data["TRAJS_POS_HIST"] = [[] for _ in range(batch_size)]  # 位置历史
        data["TRAJS_ANG_HIST"] = [[] for _ in range(batch_size)]  # 角度历史
        data["TRAJS_VEL_HIST"] = [[] for _ in range(batch_size)]  # 速度历史
        data["TRAJS_COV_HIST"] = [[] for _ in range(batch_size)]  # 协方差历史
        # 初始化元数据
        data["SCEN_PROB"] = [1.0 for _ in range(batch_size)]  # 场景概率（根节点为1）
        data["SCEN_ID"] = ["root" for _ in range(batch_size)]  # 场景ID（根节点）
        data["PARENT_ID"] = [None for _ in range(batch_size)]  # 父节点ID（无）
        data["CUR_T"] = [0 for _ in range(batch_size)]  # 当前时间（0）
        data["END_T"] = [self.pred_len for _ in range(batch_size)]  # 结束时间（预测长度）

        # 为每个样本添加历史数据
        for idx in range(batch_size):
            # 提取元素
            orig = data["ORIG"][idx]  # 原点坐标
            rot = data["ROT"][idx]  # 旋转矩阵
            trajs_ctrs = data["TRAJS"][idx]["TRAJS_CTRS"]  # 位置原点
            trajs_vecs = data["TRAJS"][idx]["TRAJS_VECS"]  # 方向向量
            theta_global = torch.atan2(rot[1, 0], rot[0, 0])  # 全局角度

            # 提取观测数据
            trajs_pos_obs = data["TRAJS"][idx]["TRAJS_POS_OBS"]  # 位置观测
            trajs_vel_obs = data["TRAJS"][idx]["TRAJS_VEL_OBS"]  # 速度观测
            trajs_ang_obs = get_angle(data["TRAJS"][idx]["TRAJS_ANG_OBS"])  # 角度观测

            # 计算每个智能体的局部坐标系变换
            trajs_theta = torch.atan2(trajs_vecs[:, 1], trajs_vecs[:, 0])  # 各智能体角度
            trajs_rots = (
                torch.stack([torch.cos(trajs_theta), -torch.sin(trajs_theta), torch.sin(trajs_theta), torch.cos(trajs_theta)], dim=1)  # 构建旋转矩阵
                .view(-1, 2, 2)
                .to(self.device)
            )

            # 轨迹历史存储变量
            trajs_pos_hist = torch.empty_like(trajs_pos_obs)  # 位置历史
            trajs_vel_hist = torch.empty_like(trajs_vel_obs)  # 速度历史
            # trajs_cov_hist = 1e-5 * torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(len(trajs_pos_obs),
            #                                                                       len(trajs_pos_obs[0]), 1, 1).to(
            #     self.device)
            # print("trajs_cov_hist: ", trajs_cov_hist.shape)

            # 初始化协方差（小值，避免零）
            trajs_cov_hist = 1e-5 * torch.ones((1,)).repeat(len(trajs_pos_obs), len(trajs_pos_obs[0]), 1).to(self.device)

            # 为每个智能体计算全局坐标系轨迹
            for i in range(len(trajs_pos_obs)):
                # 位置转换（局部到全局）
                trajs_pos_hist[i] = torch.matmul(trajs_pos_obs[i], trajs_rots[i].transpose(-1, -2)) + trajs_ctrs[i]
                # 速度转换（局部到全局）
                trajs_vel_hist[i] = torch.matmul(trajs_vel_obs[i], trajs_rots[i].transpose(-1, -2))
                # trajs_cov_hist[i] = torch.matmul(trajs_rots[i],
                #                                  torch.matmul(trajs_cov_hist[i], trajs_rots[i].transpose(-1, -2)))

            # 应用全局坐标变换
            trajs_pos_hist = torch.matmul(trajs_pos_hist, rot.T) + orig
            trajs_vel_hist = torch.matmul(trajs_vel_hist, rot.T)
            # trajs_cov_hist = torch.matmul(rot, torch.matmul(trajs_cov_hist, rot.T))
            # 角度转换：局部角度 + 智能体方向 + 全局角度
            trajs_ang_hist = trajs_ang_obs + trajs_theta.unsqueeze(1) + theta_global

            # 存储到数据字典   items in global frame
            data["TRAJS_POS_HIST"][idx] = trajs_pos_hist  # 全局位置   [N, 50, 2]
            data["TRAJS_ANG_HIST"][idx] = trajs_ang_hist  # 全局角度   [N, 50, 2]
            data["TRAJS_VEL_HIST"][idx] = trajs_vel_hist  # 全局速度   [N, 50, 2]
            data["TRAJS_COV_HIST"][idx] = trajs_cov_hist  # 协方差   [N, 50, 1]

        return data  # 返回完整数据

    def update_obser(self, cur_data):
        """更新观测数据用于下一轮预测"""
        # 提取时间信息
        end_t = cur_data["END_T"]  # 当前结束时间
        cur_t = cur_data["CUR_T"]  # 当前开始时间
        duration = end_t - cur_t  # 持续时间

        # 截取有效历史数据
        cur_data["TRAJS_POS_HIST"] = cur_data["TRAJS_POS_HIST"][:, : self.obs_len + duration]
        cur_data["TRAJS_COV_HIST"] = cur_data["TRAJS_COV_HIST"][:, : self.obs_len + duration]
        cur_data["TRAJS_ANG_HIST"] = cur_data["TRAJS_ANG_HIST"][:, : self.obs_len + duration]
        cur_data["TRAJS_VEL_HIST"] = cur_data["TRAJS_VEL_HIST"][:, : self.obs_len + duration]

        # 创建更新后的数据副本
        data = copy.deepcopy(cur_data)
        data["CUR_T"] = end_t  # 新当前时间点
        data["END_T"] = self.pred_len  # 新结束时间点

        # 截取最近的历史观测（长度为obs_len）
        data["TRAJS_POS_HIST"] = data["TRAJS_POS_HIST"][:, -self.obs_len :]
        data["TRAJS_COV_HIST"] = data["TRAJS_COV_HIST"][:, -self.obs_len :]
        data["TRAJS_ANG_HIST"] = data["TRAJS_ANG_HIST"][:, -self.obs_len :]
        data["TRAJS_VEL_HIST"] = data["TRAJS_VEL_HIST"][:, -self.obs_len :]

        # 提取轨迹数据
        trajs_pos = data["TRAJS_POS_HIST"]  # 位置
        trajs_cov = data["TRAJS_COV_HIST"]  # 协方差
        trajs_ang = data["TRAJS_ANG_HIST"]  # 角度
        trajs_vel = data["TRAJS_VEL_HIST"]  # 速度

        # 创建有效标志（假设全部有效）
        has_flags = torch.ones_like(trajs_ang)  # [N, T]

        # 提取元数据
        trajs_type = data["TRAJS"]["TRAJS_TYPE"]  # 类型
        trajs_tid = data["TRAJS"]["TRAJS_TID"]  # 轨迹ID
        trajs_cat = data["TRAJS"]["TRAJS_CAT"]  # 轨迹类别

        # ====== 坐标转换 ======
        # 计算新参考坐标系（以自车当前位置为原点）  ~ get origin and rot
        orig_seq, rot_seq, theta_seq = get_origin_rotation(trajs_pos[0], trajs_ang[0], self.device)

        # 应用全局坐标变换   ~ normalize w.r.t. scene
        trajs_pos = torch.matmul(trajs_pos - orig_seq, rot_seq)  # 位置归一化
        trajs_ang = trajs_ang - theta_seq  # 角度归一化
        trajs_vel = torch.matmul(trajs_vel, rot_seq)  # 速度归一化

        # ====== 智能体坐标系归一化 ~ normalize trajs  ======
        trajs_pos_norm = []
        trajs_ang_norm = []
        trajs_vel_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj_pos, traj_ang, traj_vel in zip(trajs_pos, trajs_ang, trajs_vel):
            # 计算每个智能体的局部坐标系
            orig_act, rot_act, theta_act = get_origin_rotation(traj_pos, traj_ang, self.device)

            # 应用坐标变换
            trajs_pos_norm.append(torch.matmul(traj_pos - orig_act, rot_act))
            trajs_ang_norm.append(traj_ang - theta_act)
            trajs_vel_norm.append(torch.matmul(traj_vel, rot_act))
            trajs_ctrs.append(orig_act)  # 位置原点
            trajs_vecs.append(torch.tensor([torch.cos(theta_act), torch.sin(theta_act)]))  # 方向向量

        # 重构为张量
        trajs_pos_obs = torch.stack(trajs_pos_norm)  # 归一化位置 [N, T, 2]  [N, 110(50), 2]
        trajs_ang_obs = torch.stack(trajs_ang_norm)  # 归一化角度 [N, T]           [N, 110(50)]
        trajs_vel_obs = torch.stack(trajs_vel_norm)  # 归一化速度 [N, T, 2]  [N, 110(50), 2]
        trajs_ctrs = torch.stack(trajs_ctrs).to(self.device)  # 位置原点 [N, 2]
        trajs_vecs = torch.stack(trajs_vecs).to(self.device)  # 方向向量 [N, 2]

        # 构建轨迹数据字典
        trajs = dict()
        # 观测部分
        trajs["TRAJS_POS_OBS"] = trajs_pos_obs  # 位置观测
        trajs["TRAJS_ANG_OBS"] = torch.stack([torch.cos(trajs_ang_obs), torch.sin(trajs_ang_obs)], dim=-1)  # 角度向量
        trajs["TRAJS_VEL_OBS"] = trajs_vel_obs  # 速度观测
        trajs["TRAJS_TYPE"] = trajs_type  # 类型
        trajs["PAD_OBS"] = has_flags[:, : self.obs_len]  # 有效标志
        # 坐标系相关   anchor ctrs & vecs
        trajs["TRAJS_CTRS"] = trajs_ctrs  # 位置原点
        trajs["TRAJS_VECS"] = trajs_vecs  # 方向向量
        # 元数据  track id & category
        trajs["TRAJS_TID"] = trajs_tid  # 轨迹ID列表
        trajs["TRAJS_CAT"] = trajs_cat  # 轨迹类别列表

        # ====== 更新车道图  ~ get lane graph ======
        lane_graph = get_new_lane_graph(self.lane_graph, orig_seq, rot_seq, self.device)

        # ====== 计算相对位置编码  ~ calc rpe ======
        rpes = dict()
        lane_ctrs = lane_graph["lane_ctrs"]  # 车道中心点
        lane_vecs = lane_graph["lane_vecs"]  # 车道方向
        scene_ctrs = torch.cat([trajs_ctrs, lane_ctrs], dim=0)  # 组合智能体和车道点
        scene_vecs = torch.cat([trajs_vecs, lane_vecs], dim=0)  # 组合智能体和车道方向
        rpes["scene"], rpes["scene_mask"] = get_rpe(scene_ctrs, scene_vecs)  # 计算场景RPE

        # ====== 更新目标点信息 ======
        tgt_pts, tgt_nodes, tgt_anch = self.get_high_level_command(orig_seq, rot_seq, trajs_vel_obs[0, -1].norm())

        # 计算目标RPE
        tgt_ctr, tgt_vec = tgt_anch
        tgt_ctrs = torch.cat([tgt_ctr.unsqueeze(0), trajs_ctrs[0].unsqueeze(0)])  # 目标点和自车原点
        tgt_vecs = torch.cat([tgt_vec.unsqueeze(0), trajs_vecs[0].unsqueeze(0)])  # 目标方向和自车方向
        tgt_rpe, _ = get_rpe(tgt_ctrs, tgt_vecs)  # 目标RPE

        # 构建完整数据
        data["ORIG"] = orig_seq  # 原点
        data["ROT"] = rot_seq  # 旋转矩阵
        data["TRAJS"] = trajs  # 轨迹数据
        data["LANE_GRAPH"] = lane_graph  # 车道图
        data["RPE"] = rpes  # 相对位置编码
        data["TGT_PTS"] = tgt_pts  # 目标点
        data["TGT_NODES"] = tgt_nodes  # 目标节点
        data["TGT_ANCH"] = tgt_anch  # 目标锚点
        data["TGT_RPE"] = tgt_rpe  # 目标相对位置编码

        return data, cur_data  # 返回新数据和原数据

    def is_condition_met(self, data):
        """检查分支终止条件（暂未实现）"""
        cov_change_rate = 9
        trajs_cov = data["TRAJS_COV_HIST"]
        cur_t = data["CUR_T"]
        end_t = data["END_T"]
        compare_t = self.obs_len + cur_t

        if cur_t == 0:
            compare_t += 1

        for t in range(cur_t + 1, end_t):
            # only check even time step
            if t % 2 == 1:
                continue

            # check if the covariance is changing too fast
            # for max sigma
            if torch.sum(trajs_cov[:, self.obs_len + t] / trajs_cov[:, compare_t] > cov_change_rate) > 0:
                data["END_T"] = t
                return False

        return True

    def get_branch_time(self, pred_data):
        """计算分支时间点（基于协方差变化率）"""
        cov_change_rate = 9  # 协方差变化率阈值
        trajs_cov = pred_data["TRAJS_COV_HIST"]  # 轨迹协方差
        cur_t = pred_data["CUR_T"]  # 当前时间
        end_t = pred_data["END_T"]  # 结束时间
        compare_t = self.obs_len + cur_t  # 参考时间点（观测结束时间点）

        # 从0开始需要补偿
        if cur_t == 0:
            compare_t += 1

        # 遍历预测时间点
        for t in range(cur_t + 1, end_t):
            # only check even time step to save computation
            # 只需检查偶数时间点
            if t % 2 == 1:
                continue

            # check if the covariance is changing too fast for max sigma
            # 检查协方差是否变化过快（最大sigma）
            if torch.any(trajs_cov[:, self.obs_len + t] / trajs_cov[:, compare_t] > cov_change_rate):
                pred_data["END_T"] = t  # 提前终止时间点
                return t  # 返回结束时间

        return end_t  # 默认返回结束时间

    def get_high_level_command(self, orig, rot, cur_vel, min_vel=0.5):
        """
        获取高级命令（目标车道点）

        返回:
            tgt_pts: 目标点序列
            tgt_nodes: 目标节点特征
            tgt_anch: 目标锚点（位置和方向）
        """
        if self.target_lane is None:
            return None, None, None

        # 计算自车到目标车道的距离  get tgt lane
        dists = torch.norm(self.target_lane - orig, dim=-1)
        # 找到最近的目标车道点   get the closest target lane point
        closest_idx = torch.argmin(dists)

        # 根据速度估算提前量  get current mind
        travel_dist = max(cur_vel, min_vel) * self.config.tar_time_ahead  # 速度*时间提前量
        # get approximation of the future area idx
        target_idx = closest_idx

        # 沿车道向前搜索
        while target_idx < len(self.target_lane) - 1 and travel_dist > 0:
            target_idx += 1
            travel_dist -= torch.norm(self.target_lane[target_idx] - self.target_lane[target_idx - 1])

        # 确保索引在有效范围内
        if target_idx == len(self.target_lane) - 1:
            target_idx -= 1

        # 选择目标点前后各5个点
        target_idx = max(5, min(target_idx, len(self.target_lane) - 6))
        selected_idx = torch.arange(target_idx - 5, target_idx + 6)

        # 提取目标车道点
        target_lane_pts = self.target_lane[selected_idx]
        target_lane_info = self.target_lane_info[selected_idx][1:]  # 移除第一个点？

        tgt_pts = copy.deepcopy(target_lane_pts)  # 目标点序列
        assert len(target_lane_pts) == 11  # 确保选择11个点

        # ====== 构建目标点特征 ======
        ctrln = copy.deepcopy(target_lane_pts)  # 车道点   [num_sub_segs + 1, 2]
        ctrln = torch.matmul(ctrln - orig, rot)  # 转换到局部坐标系  to local frame
        anch_pos = torch.mean(ctrln, dim=0)  # 计算中心点作为锚点位置
        anch_vec = (ctrln[-1] - ctrln[0]) / torch.norm(ctrln[-1] - ctrln[0])  # 计算整体方向
        anch_rot = torch.tensor([[anch_vec[0], -anch_vec[1]], [anch_vec[1], anch_vec[0]]]).to(self.device)  # 构建旋转矩阵
        ctrln = torch.matmul(ctrln - anch_pos, anch_rot)  # 转换到目标坐标系  # to instance frame

        # 计算车道段中心点和方向向量
        ctrs = (ctrln[:-1] + ctrln[1:]) / 2.0  # 段中心点
        vecs = ctrln[1:] - ctrln[:-1]  # 段方向向量
        tgt_anch = [anch_pos, anch_vec]  # 目标锚点（位置+方向）

        # 构建目标节点特征（中心点+方向向量+车道信息）
        # convert to tensor
        # ~ calc tgt feat
        tgt_nodes = torch.cat([ctrs, vecs, target_lane_info], dim=-1)  # [N_{lane}, 16, F]
        return tgt_pts, tgt_nodes, tgt_anch  # 返回结果
