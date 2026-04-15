include "map_builder.lua"
include "trajectory_builder.lua"


master_name = os.getenv("MASTER")
robot_name = os.getenv("HOST")
MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_footprint"
if (robot_name ~= "" and robot_name ~= "/") then
  robot_name = robot_name:gsub("^/", "")
  ODOM_FRAME = robot_name .. "/" .. ODOM_FRAME
  BASE_FRAME = robot_name .. "/" .. BASE_FRAME
end
if (master_name ~= "" and master_name ~= "/") then
  master_name = master_name:gsub("^/", "")
  MAP_FRAME = master_name .. "/" .. MAP_FRAME
end

options = {
  -- Map Builder logic (地图构建器算法逻辑)
  map_builder = MAP_BUILDER,
  -- Trajectory Builder logic (轨迹构建器算法逻辑)
  trajectory_builder = TRAJECTORY_BUILDER,
  
  -- The ROS frame ID of the global map (全局地图坐标系 ID)
  map_frame = MAP_FRAME,
  -- The frame tracked by SLAM; if using IMU, set to IMU frame (SLAM追踪的坐标系; 有IMU建议设为IMU系)
  tracking_frame = BASE_FRAME,
  -- The frame that Cartographer will publish poses for (算法发布位姿的目标坐标系)
  published_frame = BASE_FRAME,
  -- The frame ID for Odometry (里程计坐标系 ID)
  odom_frame = ODOM_FRAME,
  -- Whether Cartographer should provide the odom->base_link transform (是否由算法发布里程计坐标转换)
  provide_odom_frame = true,
  -- Project the published pose to 2D (remove pitch/roll) (是否将发布的位姿投影至2D)
  publish_frame_projected_to_2d = false,
  -- Predict pose using sensors for higher frequency output (使用位姿外推器进行高频预测输出)
  use_pose_extrapolator = true,
  -- Use odometry data as an additional input (是否使用里程计数据)
  use_odometry = true,
  -- Use GPS/NavSat data (是否使用导航卫星定位数据)
  use_nav_sat = false,
  -- Use Landmark data (是否使用路标点数据)
  use_landmarks = false,
  
  -- Number of single-echo LiDAR topics (单线激光雷达话题数量)
  num_laser_scans = 1,
  -- Number of multi-echo LiDAR topics (多回波激光雷达话题数量)
  num_multi_echo_laser_scans = 0,
  -- Number of subdivisions per scan (每帧雷达数据的切割细分数)
  num_subdivisions_per_laser_scan = 1,
  -- Number of 3D PointCloud2 topics (3D点云话题数量)
  num_point_clouds = 0,
  -- Timeout for looking up TF transforms (查找TF坐标变换的超时时间)
  lookup_transform_timeout_sec = 0.2,
  -- Period to publish submap updates to ROS (发布子图更新的周期)
  submap_publish_period_sec = 0.3,
  -- Period to publish pose to ROS (发布位姿的话题周期)
  pose_publish_period_sec = 5e-3,
  -- Period to publish trajectory (发布轨迹的周期)
  trajectory_publish_period_sec = 30e-3,
  
  -- Ratio of rangefinder data to use (激光数据采样比例)
  rangefinder_sampling_ratio = 0.5,
  -- Ratio of odometry data to use (里程计数据采样比例)
  odometry_sampling_ratio = 1.0,
  -- Ratio of fixed frame poses to use (固定帧位姿采样比例)
  fixed_frame_pose_sampling_ratio = 1.,
  -- Ratio of IMU data to use (IMU数据采样比例)
  imu_sampling_ratio = 1.0,
  -- Ratio of landmarks to use (路标点采样比例)
  landmarks_sampling_ratio = 1.,
}

-- Enable 2D SLAM (启用2D建图)
MAP_BUILDER.use_trajectory_builder_2d = true

-- Number of scan data per submap (每个子图包含的雷达扫描数)
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 60


-- Minimum range for LiDAR data (雷达最小有效距离)
TRAJECTORY_BUILDER_2D.min_range = 0.1
-- Maximum range for LiDAR data (雷达最大有效距离)
TRAJECTORY_BUILDER_2D.max_range = 7.
-- Ray length for missing LiDAR data (缺失雷达数据时的光线填充长度)
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.
-- Whether to use IMU data in the 2D trajectory builder (2D前端是否使用IMU)
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- Use correlative scan matching for better initial guesses (使用相关扫描匹配以获得更好的初始位姿预测)
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
-- Search window for real-time scan matching (实时扫描匹配的平移搜索窗口)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
-- Penalty weight for translation deviation (平移偏差的惩罚权重)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 30.
-- Penalty weight for rotation deviation (旋转偏差的惩罚权重)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1.0


-- Scale of the Huber loss function for outlier rejection (Huber损失函数比例，用于剔除异常值)
POSE_GRAPH.optimization_problem.huber_scale = 1e2
-- Optimize the global pose graph every N nodes (每增加N个节点执行一次全局优化/回环检测)
POSE_GRAPH.optimize_every_n_nodes = 120
-- Minimum score for a constraint to be considered valid (约束建立/回环检测的最低评分限制)
POSE_GRAPH.constraint_builder.min_score = 0.85

return options