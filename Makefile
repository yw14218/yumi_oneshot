.PHONY: launch_camera launch_handeye launch_yumi launch_moveit stop all

all: launch_all

camera:
	@$(MAKE) launch_camera

yumi: launch_yumi

moveit: launch_moveit

bridge:launch_bridge

launch_all:
	@echo "Setting network configuration for Yumi..."
	@sudo ifconfig eno1 192.168.125.50
	@echo "Launching Yumi Trajectory Position Control..."
	@gnome-terminal -- bash -c "roslaunch yumi_launch yumi_traj_pos_control.launch 2> >(grep -v TF_REPEATED_DATA)"
	@sleep 1
	@echo "Launching Yumi MoveIt! Configuration..."
	@gnome-terminal -- bash -c "roslaunch yumi_moveit_config demo_online.launch 2> >(grep -v TF_REPEATED_DATA)"
	@sleep 1
	@echo "Launching ROS1 Bridge..."
	@gnome-terminal -- bash -c "source /opt/ros/foxy/setup.bash && ros2 run ros1_bridge dynamic_bridge --bridge-all-topics"
	@sleep 1
	@echo "Launching RealSense Camera..."
	@gnome-terminal -- bash -c "source /opt/ros/foxy/setup.bash && source /opt/ros/foxy/local_setup.bash && \
	 (ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true device_type:=d405 camera_name:=d405 &) && \
	 (ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true device_type:=d415 camera_name:=d415)"

launch_bridge:
	@echo "Launching ROS1 Bridge..."
	@bash -c "source /opt/ros/foxy/setup.bash && ros2 run ros1_bridge dynamic_bridge --bridge-all-topics"

launch_camera:
	@echo "Launching RealSense Camera..."
	@bash -c "source /opt/ros/foxy/setup.bash && source /opt/ros/foxy/local_setup.bash && \
	 (ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true device_type:=d405 camera_name:=d405 &) && \
	 (ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true device_type:=d415 camera_name:=d415)"

launch_d415:
	@echo "Launching RealSense Camera..."
	@bash -c "source /opt/ros/foxy/setup.bash && source /opt/ros/foxy/local_setup.bash && \
	 (ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true device_type:=d415 camera_name:=d415)"

launch_d405:
	@echo "Launching RealSense Camera..."
	@bash -c "source /opt/ros/foxy/setup.bash && source /opt/ros/foxy/local_setup.bash && \
	 (ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true device_type:=d405 camera_name:=d405)"

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
	@pkill -f ros


# source /opt/ros/foxy/setup.bash && ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
# ros2 launch realsense2_camera rs_multi_camera_launch.py camera_name1:=D415 camera_name2:=D405 device_type1:=d415 device_type2:=d405 pointcloud.enable:=true
# ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true device_type:=d405 camera_name:=d405
# ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true device_type:=d415 camera_name:=d415
