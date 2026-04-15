# ROSOrin Pro

[English](README.md) | 中文

<p align="center">
  面向多模态感知、导航与具身智能开发的 ROS2 智能小车平台
</p>

## 产品概述

### 关于 ROSOrin Pro

ROSOrin Pro 是我们面向机器人教育、具身智能开发和真实场景部署打造的 ROS2 智能小车平台。平台集成高性能主控、STM32 机器人控制器、激光雷达、3D 视觉和机械臂执行能力，为开发者提供了从基础控制到高级自主交互的一体化实践路径。

当前目录提供的是 ROSOrin Pro 在整机部署流程中使用的控制器侧发布内容。它为整个平台提供稳定的底层执行基础，并与 ROSOrin Pro 更完整的软件与硬件生态保持一致。

无论你是在做教学平台、实验室原型，还是更复杂的机器人应用演示，ROSOrin Pro 都能为运动控制、外设协同和整机执行提供清晰而可靠的起点。

### 核心：为真实机器人执行打造的双控制器架构

ROSOrin Pro 采用分层硬件架构。上位机负责感知、规划与高层交互，STM32 控制器负责电机、舵机以及外设的实时控制。这样的设计可以兼顾响应速度、系统稳定性和后续扩展能力。

当前目录中的控制器侧内容，正是这套架构中的执行核心。它承担移动底盘控制、机械臂动作协同以及板载外设联动所需的底层控制任务。

### 软件生态：为整机工作流提供稳定底座

虽然当前目录结构较精简，但它在 ROSOrin Pro 整体工作流中承担着关键角色。无论是建图导航、语音交互、视觉追踪、自主搬运还是抓取执行，上层能力都建立在这套稳定的控制器基础之上。

通过将控制层保持得足够清晰和专注，ROSOrin Pro 让开发者能够把更多精力放在上层 ROS2 应用开发上，同时保持真实机器人执行过程的稳定性。

### 你能做什么

基于 ROSOrin Pro，你可以构建一整套完整的机器人应用链路，包括：

**底盘运动控制** — 为移动机器人提供稳定的底层驱动和协调动作基础。

**外设协同控制** — 支持舵机、传感器及其他板载模块的可靠联动。

**整机执行支撑** — 为导航、跟踪、抓取、搬运等上层任务提供可信的控制底座。

**系统级扩展集成** — 让控制层自然接入更完整的 ROSOrin Pro 软件生态和后续扩展方案。

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
- 电源及配套线材

### 软件环境搭建

1. 将 `STM32/` 目录中的控制器固件烧录到机器人控制板。
2. 将控制板接入对应的 ROSOrin Pro 硬件平台。
3. 其余整机部署和上位机环境搭建，请参考官方文档完成。

完整安装与部署流程请参考[官方文档](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/)。

## 仓库结构

```
ROSOrin Pro/
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
