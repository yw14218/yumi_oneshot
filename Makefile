.PHONY: launch_camera launch_handeye launch_yumi launch_moveit stop all

all: launch_camera launch_handeye launch_yumi launch_moveit
camera:
	@$(MAKE) launch_camera &
	@$(MAKE) launch_handeye

yumi: launch_yumi
moveit: launch_moveit

launch_camera:
	@echo "Launching RealSense Camera..."
	@roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud &

launch_handeye:
	@echo "Launching Yumi Handeye Calibration..."
	@sleep 1 # Wait for the camera to initialize
	@bash -c "roslaunch handeye/handeye.launch"

launch_yumi:
	@echo "Setting network configuration for Yumi..."
	@sudo ifconfig eno1 192.168.125.50
	@echo "Launching Yumi Trajectory Position Control..."
	@bash -c "roslaunch yumi_launch yumi_traj_pos_control.launch 2> >(grep -v TF_REPEATED_DATA)"

launch_moveit:
	@echo "Launching Yumi MoveIt! Configuration..."
	@bash -c "roslaunch yumi_moveit_config demo_online.launch 2> >(grep -v TF_REPEATED_DATA)"

stop:
	@echo "Stopping all ROS nodes..."
	@killall -q roslaunch || true
	@killall -q rosmaster || true