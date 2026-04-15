# ROSOrin Pro

English | [中文](README_cn.md)

<p align="center">
  ROS2 Smart Car Platform for Multimodal Perception, Navigation, and Embodied AI
</p>

## Product Overview

### About ROSOrin Pro

ROSOrin Pro is our ROS2 smart car platform built for robotics education, embodied AI development, and advanced autonomous applications. By combining a high-performance host controller, an STM32 robot controller, LiDAR, 3D vision, voice interaction, and robotic arm execution, ROSOrin Pro creates a complete workflow from perception and planning to physical action.

This package includes a full ROS2 workspace together with controller firmware, making it suitable for developers who want to move from system bringup to mapping, navigation, tracking, manipulation, and multimodal task execution within one integrated platform.

Whether you are exploring SLAM, autonomous driving, intelligent grasping, or embodied AI interaction, ROSOrin Pro provides a structured and expandable foundation for real robot development.

### The Core: A Multimodal Robot Platform Built Around ROS2

ROSOrin Pro adopts a dual-controller architecture. The upper computer handles perception, planning, and application logic, while the STM32 controller manages real-time robot execution. This architecture keeps motion control responsive while allowing the platform to support richer AI and robotics workloads.

At the hardware level, ROSOrin Pro combines mobile chassis mobility with LiDAR, a depth camera, and robotic arm capability, enabling the platform to work across both navigation and manipulation scenarios.

### The Software Stack: From Bringup to Autonomous Tasks

The workspace includes packages for bringup, calibration, drivers, navigation, SLAM, peripherals, simulation, multimodal interaction, large-model demos, and example applications. Core capabilities in the codebase include line following, object tracking, self-driving workflows, voice interaction, visual perception, inverse kinematics, and multi-robot orchestration.

The platform also integrates widely used robotics and vision components such as OpenCV, MediaPipe, YOLO-based detection, KCF tracking, Gmapping, and ROS2 launch-based modular deployment, giving developers a practical stack for rapid experimentation and expansion.

### What You Can Build

A complete project path is available across control, perception, and embodied interaction:

**SLAM and Navigation** — Build maps, localize the robot, plan paths, and run autonomous transport workflows in structured environments.

**Vision and Tracking** — Develop line following, object tracking, depth-based interaction, gesture recognition, and target-aware applications.

**Voice and Multimodal Interaction** — Combine voice control, visual perception, and task execution for more natural robot interaction.

**Manipulation and Transport** — Coordinate mobile movement with robotic arm action for grasping, sorting, and transport tasks.

**Simulation and Expansion** — Use the provided simulation, driver, and example packages as a base for secondary development and custom robotics projects.

## Official Resources

### Official Hiwonder

- **Official Website**: [https://www.hiwonder.com/](https://www.hiwonder.com/)
- **Product Page**: [https://www.hiwonder.com/products/rosorin-pro](https://www.hiwonder.com/products/rosorin-pro)
- **Official Documentation**: [https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/)
- **Technical Support**: support@hiwonder.com

## Getting Started

### Hardware Requirements

- ROSOrin Pro hardware platform
- STM32 robot controller
- LiDAR
- 3D depth camera
- Power supply and required wiring

### Software Setup

1. Enter the `ROS2/src_ros2/` workspace and complete the ROS2 build process.
2. Flash the controller firmware in the `STM32/` directory to the robot controller board.
3. Configure the runtime environment according to the official deployment guide.
4. Launch the required application modules from the corresponding ROS2 packages and launch files.

Refer to the [official documentation](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/) for the complete installation and deployment process.

## Repository Structure

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

## Community & Support

- **GitHub Issues**: Report bugs and request features
- **Email Support**: support@hiwonder.com
- **Documentation**: Comprehensive guides and tutorials

## License

This project is open-source and available for educational and research purposes.

---

**Hiwonder** - Empowering Innovation in Robotics Education
