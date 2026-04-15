# ROSOrin Pro

[English](README.md) | 中文

<p align="center">
  面向多模态感知、导航与具身智能开发的 ROS2 智能小车平台
</p>

## 产品概述

### 关于 ROSOrin Pro

ROSOrin Pro 是我们面向机器人教育、具身智能开发和高级自主应用打造的 ROS2 智能小车平台。平台将高性能主控、STM32 机器人控制器、激光雷达、3D 视觉、语音交互和机械臂执行能力整合在一起，形成从感知、规划到物理执行的一体化开发链路。

当前目录同时包含完整的 ROS2 工作空间和控制器固件，适合希望从系统启动、建图导航一路走到目标跟踪、机械臂操作和多模态任务执行的开发者使用。

无论你是在做 SLAM、自主驾驶、智能抓取还是具身智能交互，ROSOrin Pro 都能提供一套结构清晰、便于扩展的真实机器人开发基础。

### 核心：围绕 ROS2 构建的多模态机器人平台

ROSOrin Pro 采用双控制器架构。上位机负责感知、规划和应用逻辑，STM32 控制器负责整机实时执行。这样的设计既保证了运动控制的响应速度，也让平台能够承载更丰富的 AI 与机器人应用能力。

在硬件层面，ROSOrin Pro 将移动底盘、激光雷达、深度相机和机械臂整合在同一平台上，让机器人既能完成导航任务，也能承担抓取与搬运等更复杂的执行场景。

### 软件生态：从启动部署到自主任务

工作空间中包含 bringup、calibration、driver、navigation、slam、peripherals、simulation、多模态交互、大模型演示以及示例应用等模块。代码中已经覆盖巡线、目标跟踪、自动驾驶流程、语音交互、视觉感知、逆运动学和多机协同等核心能力。

平台同时整合了 OpenCV、MediaPipe、YOLO 检测、KCF 跟踪、Gmapping 以及基于 ROS2 launch 的模块化部署方式，为开发者提供了一套上手快、扩展性强的机器人应用栈。

### 你能做什么

围绕控制、感知和具身交互，ROSOrin Pro 可以支持完整的项目开发路径：

**SLAM 与导航** — 完成建图、定位、路径规划以及结构化场景下的自主搬运任务。

**视觉与跟踪** — 实现巡线、目标跟踪、深度交互、手势识别等视觉类应用。

**语音与多模态交互** — 将语音控制、视觉感知和任务执行结合起来，打造更自然的机器人交互体验。

**抓取与搬运** — 联动移动底盘和机械臂能力，完成抓取、分拣和搬运等操作任务。

**仿真与二次开发** — 基于现有仿真、驱动与示例工程，快速扩展自己的机器人项目。

## 官方资源

### 幻尔科技官方

- **官方网站**: [https://www.hiwonder.com/](https://www.hiwonder.com/)
- **产品页面**: [https://www.hiwonder.com/products/rosorin-pro](https://www.hiwonder.com/products/rosorin-pro)
- **官方文档**: [https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/)
- **技术支持**: support@hiwonder.com

## 快速开始

### 硬件准备

- ROSOrin Pro 机器人平台
- STM32 机器人控制器
- 激光雷达
- 3D 深度相机
- 电源及配套线材

### 软件环境搭建

1. 进入 `ROS2/src_ros2/` 工作空间并完成 ROS2 编译。
2. 将 `STM32/` 目录中的控制器固件烧录到机器人控制板。
3. 按官方部署指南完成运行环境配置。
4. 根据需要，从对应的 ROS2 包和 launch 文件启动功能模块。

完整安装与部署流程请参考[官方文档](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/)。

## 仓库结构

```
ROSOrin Pro/
├── ROS2/
│   └── src_ros2/
│       ├── src/
│       │   ├── app/
│       │   ├── bringup/
│       │   ├── calibration/
│       │   ├── driver/
│       │   ├── example/
│       │   ├── large_models/
│       │   ├── multi/
│       │   ├── navigation/
│       │   ├── peripherals/
│       │   ├── simulations/
│       │   ├── slam/
│       │   └── ...
└── STM32/
    └── RosRobotControllerM4-20260414.hex
```

## 社区与支持

- **GitHub Issues**: 提交问题反馈和功能建议
- **邮件支持**: support@hiwonder.com
- **文档资料**: 完整的教程与开发指南

## 许可证

本项目开源，可用于教育和研究目的。

---

**幻尔科技** - 赋能机器人教育创新
