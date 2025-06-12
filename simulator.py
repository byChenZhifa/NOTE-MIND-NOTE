from datetime import datetime
import os
import json
import shutil
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 可视化工具函数
from common.visualization import draw_map, draw_agent, draw_scen_trees, reset_ax, draw_traj_trees, draw_traj
from agent import CustomizedAgent, NonReactiveAgent  # 智能体实现
from loader import ArgoAgentLoader
from common.semantic_map import SemanticMap

matplotlib.use("Agg")


class Simulator:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = json.load(file)  # 加载JSON格式的配置文件

        # 从配置中获取仿真名称和序列ID
        self.sim_name = self.config["sim_name"]
        self.seq_id = self.config["seq_id"]

        #  output_dir 创建输出目录
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_configs = os.path.abspath(os.path.dirname(config_path))  # NOTE-MIND-NOTE/configs
        dir_MIND_root = os.path.abspath(os.path.dirname(dir_configs))  # NOTE-MIND-NOTE/
        self.seq_path = os.path.join(dir_MIND_root, "data", self.seq_id)  # NOTE-MIND-NOTE/data/xxxx
        self.output_dir = os.path.join(dir_MIND_root, "outputs", self.sim_name, f"{self.sim_name}_{time_str}")  # NOTE-MIND-NOTE/outputs/xxxx

        # 仿真参数配置
        # self.output_dir = self.config['output_dir']
        self.num_threads = self.config["num_threads"]  # 线程数量
        # self.seq_path = os.path.join('data/', self.seq_id)
        self.map_dir_argo2 = os.path.abspath(os.path.join(self.seq_path, f"log_map_archive_{self.seq_id}.json"))  # Argo地图路径

        # 初始化语义地图
        self.smp = SemanticMap()
        self.smp.load_from_argo2(Path(self.map_dir_argo2))  # 从Argo格式加载地图数据

        # 渲染和智能体控制相关参数
        self.render = self.config["render"]  # 是否渲染可视化
        self.cl_agents = self.config["cl_agents"]  # 控制智能体配置

        # 仿真时序参数
        self.sim_time = 0.0  # 当前仿真时间（秒）
        self.sim_step = 0.02  # 仿真步长（秒）
        self.sim_horizon = 500  # 最大仿真步数

        # 存储数据结构初始化
        self.agents = []  # 所有智能体对象列表
        self.frames = []  # 存储每帧数据用于渲染

    def run(self):
        """主运行函数，包含整个仿真流程"""
        self.init_sim()  # 初始化仿真环境
        self.run_sim()  # 执行仿真主循环
        self.render_video()  # 渲染结果视频

    def init_sim(self):
        """初始化仿真环境"""
        self.agents = []  # 清空智能体列表

        # 加载场景数据
        scenario_path = Path(self.seq_path + f"/scenario_{self.seq_id}.parquet")  # 场景文件路径
        replay_agent_loader = ArgoAgentLoader(scenario_path)  # 创建场景加载器

        # 加载控制智能体
        self.agents += replay_agent_loader.load_agents(self.smp, self.cl_agents)  # 从场景文件加载智能体
        aaa = 1

    def run_sim(self):
        """执行仿真主循环"""
        print("Running simulation...")
        # 重置仿真时间和帧数据   # reset sim time and frames
        self.frames = []
        self.sim_time = 0.0
        terminated = False  # 提前终止标志

        # 主仿真循环（带进度条）
        for sim_horizon_index in tqdm(range(self.sim_horizon)):
            frame_agents_len = 0  # 当前帧中的智能体数量
            frame = {}  # 创建当前帧数据存储结构

            # ========= 获取智能体观测 =========
            agent_obs = []  # 当前所有智能体的观测数据（带噪声）
            for agent in self.agents:
                # 只获取有效非反应性智能体和可控智能体的观测
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_obs.append(agent.observe())  # 获取带噪声的观测数据

            # ========= 获取真实状态（无噪声）用于记录 =========
            agent_gt = []  # 真实状态（用于可视化）
            for agent in self.agents:
                # 只获取有效非反应性智能体和可控智能体的真实状态
                if (isinstance(agent, NonReactiveAgent) and agent.is_valid()) or isinstance(agent, CustomizedAgent):
                    agent_gt.append(agent.observe_no_noise())  # 获取无噪声的真实状态

            # 记录真实状态到当前帧
            frame["agents"] = agent_gt
            frame_agents_len += len(agent_gt)  # 更新帧中智能体计数

            # ========= 智能体决策与状态更新 =========
            # Update local semantic map and plan; 依次往前单步回放 agent;
            for agent_index, agent in enumerate(self.agents):
                if isinstance(agent, CustomizedAgent):  # 处理可规划智能体（如自车）
                    agent.check_enable(self.sim_time)  # 检查是否启用规划
                    # 检查记录和规划触发条件
                    rec_tri, pl_tri = agent.check_trigger(self.sim_time)  # record_trigger, planner_trigger  #! 0.1s触发一次规划

                    if rec_tri:  # 记录触发
                        agent.step()  # 推进记录索引

                    if pl_tri:  # 规划触发
                        agent.update_observation(agent_obs)  # 更新智能体观测

                        if agent.is_enable:  # 如果启用则进行规划
                            is_success, res = agent.plan()  # 执行规划算法
                            if not is_success:  # 规划失败处理
                                print("Agent {} plan failed!".format(agent.id))
                                terminated = True  # 设置提前终止标志
                                break  # 退出当前智能体处理循环
                            aget_av_paln = is_success

                            # 记录自车的规划结果（用于可视化）
                            if agent.id == "AV":
                                frame["scen_tree"] = res[0]  # 场景树
                                frame["traj_tree"] = res[1]  # 轨迹树
                            aget_av = 1
                        pass  # if agent.is_enable:
                    pass_aa1 = 1

                elif isinstance(agent, NonReactiveAgent):  # 处理非反应性智能体（如背景车）
                    agent.step()  # 推进记录索引

                else:  # 未知智能体类型处理
                    raise ValueError("Unknown agent type")

                # 所有智能体更新状态
                agent.update_state(self.sim_step)  # 应用动力学模型更新状态
                pass  # if isinstance(agent, CustomizedAgent):  elif  else END
                aaaaa = 1  #  for agent in self.agents:
            pass  # for agent in self.agents:

            # ========= 保存当前帧并推进时间 =========
            self.frames.append(frame)  # 存储当前帧数据
            self.sim_time += self.sim_step  # 时间推进
            if self.sim_time > 4.30:
                sssss = 1

            # 检查是否达到提前终止条件
            if terminated:
                print("Simulation terminated!")
                break  # 退出仿真主循环

    def render_video(self):
        """渲染结果视频"""
        if not self.render:  # 如果不需要渲染则直接返回
            return
        print("Rendering video...")
        # 确保输出目录存在  # check directory exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 创建图像保存目录
        img_dir = self.output_dir + "/imgs"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 多线程渲染每一帧
        # output frame png multiprocessing use the spawn method to create a new process
        # 使用多进程并行渲染（使用spawn方式创建进程）
        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(self.num_threads)

        # 为每一帧图像创建渲染任务
        pool.starmap(self.render_png, [(frame_idx, img_dir) for frame_idx in range(len(self.frames))])
        pool.close()  # 关闭进程池
        pool.join()  # 等待所有进程完成

        # call ffmpeg to combine images into a video
        # video_name = f"{self.seq_id}_{self.sim_name}.mov"
        # output_command = "ffmpeg -r 25 -i " + img_dir + f"/frame_%03d.png" + " -vcodec mpeg4 -y " + self.output_dir + video_name
        # os.system(output_command)
        # shutil.rmtree(img_dir) # 这行 代码会删除图片目录

    ########################################
    # Visualization functions
    ########################################
    def render_png(self, frame_idx, img_dir):
        """渲染单帧图像并保存为PNG文件"""
        fig = plt.figure(figsize=(48, 48))  # 创建大尺寸画布
        ax = fig.add_subplot(111, projection="3d")  # 创建3D坐标轴
        plt.tight_layout()  # 紧凑布局
        self.render_frame(frame_idx, ax)  # 在坐标轴上渲染当前帧

        # Save the frame with directory path  保存PNG文件
        frame_filename = img_dir + f"/frame_{frame_idx:03d}.png"
        plt.tight_layout()
        plt.savefig(frame_filename)
        plt.close(fig)  # 关闭图形释放内存

    def render_frame(self, frame_idx, ax):
        """在指定坐标轴上渲染单个仿真帧"""
        # 尝试获取场景树可视化数据（从当前或之前的帧）
        scen_tree_vis = None

        # retrieve the vis data from the previous frame to avoid the empty visualization
        if "scen_tree" in self.frames[frame_idx]:
            scen_tree_vis = self.frames[frame_idx]["scen_tree"]
        else:  # 如果当前帧没有场景树数据，查找最近的可用帧
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and "scen_tree" not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and "scen_tree" in self.frames[pre_frame_idx]:
                scen_tree_vis = self.frames[pre_frame_idx]["scen_tree"]

        # 尝试获取轨迹树可视化数据（从当前或之前的帧）
        traj_tree_vis = None
        if "traj_tree" in self.frames[frame_idx]:
            traj_tree_vis = self.frames[frame_idx]["traj_tree"]
        else:  # 如果当前帧没有轨迹树数据，查找最近的可用帧
            pre_frame_idx = frame_idx - 1
            while pre_frame_idx >= 0 and "traj_tree" not in self.frames[pre_frame_idx]:
                pre_frame_idx -= 1
            if pre_frame_idx >= 0 and "traj_tree" in self.frames[pre_frame_idx]:
                traj_tree_vis = self.frames[pre_frame_idx]["traj_tree"]

        # Clear the previous cube and draw a new one  准备3D可视化空间
        range_3d = 15.0  # 可视范围半径
        font_size = 35  # 文本字体大小
        reset_ax(ax)  # 重置坐标轴

        # Process the frame   配置相机视角
        center = np.array([0, 0])  # 场景中心点
        center[0] = self.config["render_config"]["camera_position"]["x"]
        center[1] = self.config["render_config"]["camera_position"]["y"]
        cam_yaw = self.config["render_config"]["camera_position"]["yaw"]
        elev = self.config["render_config"]["camera_position"]["elev"]

        # 设置坐标轴范围
        ax.set_xlim([center[0] - range_3d, center[0] + range_3d])
        ax.set_ylim([center[1] - range_3d, center[1] + range_3d])
        ax.set_zlim([0, 2 * range_3d])
        ax.view_init(elev=elev, azim=180 + np.rad2deg(cam_yaw))  # 设置3D视角

        # ========= 绘制地图元素 =========
        draw_map(ax, self.smp.map_data)  # 绘制语义地图

        # ========= 绘制规划树 =========
        if scen_tree_vis is not None:
            draw_scen_trees(ax, scen_tree_vis)  # 绘制场景树
        if traj_tree_vis is not None:
            draw_traj_trees(ax, traj_tree_vis)  # 绘制轨迹树

        # ========= 绘制智能体 =========
        #  plot agents
        for agent in self.frames[frame_idx]["agents"]:
            draw_agent(ax, agent, vis_bbox=False)  # 绘制智能体状态

            # 在智能体位置添加速度和ID标签
            if np.linalg.norm(agent.state[:2] - center) < 2 * range_3d:
                ax.text(agent.state[0], agent.state[1], 1.0, "No.{}:{:.2f}m/s".format(agent.id, agent.state[2]), fontsize=font_size)

        # ========= 绘制轨迹历史 =========
        # try to retrieve the history of the agent in current frame
        agent_history = dict()  # 存储每个智能体的历史轨迹点

        # 初始化当前帧智能体历史
        for agent in self.frames[frame_idx]["agents"]:
            agent_history[agent.id] = [agent.state[:2]]  # 当前位置作为第一个点

        # 回溯历史帧获取轨迹点
        back_step = 100  # 最大回溯步数
        for i in range(1, back_step):
            if frame_idx - i < 0:  # 避免越界
                break
            # 遍历历史帧中的智能体
            for agent in self.frames[frame_idx - i]["agents"]:
                if agent.id in agent_history:  # 只收集还在当前视图的智能体
                    agent_history[agent.id].append(agent.state[:2])  # 添加历史位置

        # 绘制每个智能体的历史轨迹 plot the history of the agent
        for agent_id, history in agent_history.items():
            history.reverse()  # 反转列表使新点在最后

            # 跳过静止或几乎静止的轨迹
            # check length of history
            if np.linalg.norm(history[0] - history[-1]) < 0.1:
                continue
            draw_traj(ax, history)  # 绘制轨迹
