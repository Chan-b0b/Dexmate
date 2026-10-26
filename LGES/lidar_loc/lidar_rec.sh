#!/usr/bin/env bash
# Log-only lidar recording next to a normal chassis_sequence run (commands nothing).
#   ./lidar_rec.sh start   # lidar stage only (no robot/localization/nav) + 1 Hz recorder
#   ./lidar_rec.sh stop    # stop the recorder, copy frames to ../ik_demo/run_logs/lidar/<stamp>
# The lidar stage keeps running after stop (harmless); `./lidar_rec.sh lidar-off` stops it.
set -e
C=lgcns_nav2_apriltag
HERE=$(cd "$(dirname "$0")" && pwd)
ROS='eval "$(grep -hE "^export (ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION)=" /root/.bashrc)"; source /opt/ros/humble/setup.bash'
case "$1" in
  start)
    if ! docker exec $C bash -c "$ROS; ros2 node list" | grep -q rslidar_sdk_node; then
      docker exec -d $C bash -c "$ROS; source /root/ros_ws/navigation/custom_nav/install/local_setup.bash; exec ros2 launch vega_nav_bringup vega_full_launch.py enable_robot:=false enable_lidar:=true enable_perception:=false enable_localization:=false enable_nav:=false use_rviz:=false > /tmp/lidar_only.log 2>&1"
      echo "lidar stage launched (lidar only)"; sleep 6
    fi
    STAMP=$(date +%Y%m%d_%H%M%S)
    docker cp "$HERE/lidar_rec.py" $C:/tmp/lidar_rec.py
    docker exec -d $C bash -c "$ROS; exec python3 /tmp/lidar_rec.py /tmp/lidar_rec_$STAMP 7200 > /tmp/lidar_rec.log 2>&1"
    echo "recording -> (container) /tmp/lidar_rec_$STAMP (auto-stops after 2 h)";;
  stop)
    docker exec $C pkill -f lidar_rec.py || true
    DIR=$(docker exec $C bash -c 'ls -d /tmp/lidar_rec_2* | tail -1')
    mkdir -p "$HERE/../ik_demo/run_logs/lidar"
    docker cp $C:$DIR "$HERE/../ik_demo/run_logs/lidar/" && docker exec $C rm -rf "$DIR"
    echo "copied $(ls "$HERE/../ik_demo/run_logs/lidar/$(basename $DIR)" | wc -l) frames -> ik_demo/run_logs/lidar/$(basename $DIR)";;
  lidar-off)
    docker exec $C pkill -f "vega_full_launch.py" || true; echo "lidar stage stopped";;
  *) echo "usage: $0 start|stop|lidar-off"; exit 1;;
esac
