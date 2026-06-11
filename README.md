# ROSOrin Pro

English | [中文](README_cn.md)

<p align="center">
  ROS2 Smart Car Platform for Multimodal Perception, Navigation, and Embodied AI
</p>

## Product Overview

### About ROSOrin Pro

ROSOrin Pro is our ROS2 smart car platform built for robotics education, embodied AI development, and real-world robot deployment. By combining a high-performance host controller, an STM32 robot controller, LiDAR, 3D vision, and robotic arm execution, ROSOrin Pro provides a practical path from foundational robot control to advanced autonomous interaction.

This package delivers the controller-side release assets used in the ROSOrin Pro deployment workflow. It is designed to give developers a stable robot-side control foundation while staying aligned with the broader ROSOrin Pro software and hardware ecosystem.

Whether you are building a teaching platform, a lab prototype, or an advanced robotics demo, ROSOrin Pro offers a structured foundation for motion control, peripheral coordination, and integrated robot execution.

### The Core: A Dual-Controller Architecture Built for Reliable Robot Execution

ROSOrin Pro adopts a layered hardware architecture. The upper computer is responsible for perception, planning, and high-level interaction, while the STM32 controller focuses on real-time motor control, servo coordination, and peripheral response. This architecture improves responsiveness, keeps control logic stable, and makes the platform easier to scale into more complex applications.

The controller-side package in this directory is the execution anchor of that architecture. It supports the low-level coordination required by mobile chassis control, robotic arm motion, and onboard peripheral interaction.

### The Software Stack: A Stable Foundation for Integrated Robot Workflows

Although lightweight in structure, this package plays an essential role in the ROSOrin Pro workflow. It serves as the firmware layer beneath mapping, navigation, voice interaction, visual tracking, autonomous transport, and manipulation tasks.

By keeping the controller layer clear and focused, ROSOrin Pro allows developers to move faster on upper-level ROS2 applications while maintaining reliable real-world execution at the hardware layer.

### What You Can Build

With ROSOrin Pro, you can build a complete robotics workflow that spans:

**Chassis Motion Control** — Stable low-level control for mobile robot movement and coordinated driving behavior.

**Peripheral Coordination** — Reliable execution for servos, sensors, and other onboard hardware modules.

**Robot-Side Execution** — A dependable controller foundation for navigation, tracking, grasping, transport, and other higher-level tasks.

**Scalable System Integration** — A hardware control layer that fits naturally into the wider ROSOrin Pro ecosystem and future application expansion.

## Official Resources

### Official Hiwonder

- **Official Website**: [https://www.hiwonder.com/](https://www.hiwonder.com/)
- **Product Page**: [https://www.hiwonder.com/products/rosorin-pro](https://www.hiwonder.com/products/rosorin-pro)
- **Official Documentation**: [https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/](https://wiki.hiwonder.com/projects/rosorin-pro/en/latest/)
- **Technical Support**: support@hiwonder.com

## Getting Started

### Hardware Requirements

- ROSOrin Pro hardware platform
- STM32 robot controller
- Power supply and required wiring

### Software Setup

1. Flash the controller firmware in the `STM32/` directory to the robot controller board.
2. Connect the controller board to the corresponding ROSOrin Pro hardware platform.
3. Complete the rest of the hardware deployment and host-side setup by following the official documentation.

Refer to the [official documentation](https://docs.hiwonder.com/projects/ROSOrin/en/jetson-orin-nano-version/) for the complete installation and deployment process.

## Repository Structure

```
ROSOrin Pro/
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
