import numpy as np
import tyro

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

import threading
import time
from queue import Queue
import json

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
#from geometry_msgs.msg import Point

from utils import align_head_to_forward


### transformatation functions
def _rot_y(theta: float) -> np.ndarray:
    """Rotation matrix around Y axis by theta (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    """Rotation matrix around Z axis by theta (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ], dtype=np.float64)
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _trans(x: float, y: float, z: float) -> np.ndarray:
    """4x4 translation matrix."""
    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _T_base_arm_center(q_torso: np.ndarray) -> np.ndarray:
    """Base frame에서 arm_center까지 4x4 변환 (torso 관절각만 사용)."""
    q = np.asarray(q_torso, dtype=np.float64).ravel()[:3]
    q1, q2, q3 = q[0], q[1], q[2]
    
    # Joint frames (vega_1p URDF)
    T_0_1 = _trans(-0.235, 0.0, 0.248) @ _rot_y(-q1)
    T_1_2 = _trans(0.396, 0.0, 0.082) @ _rot_y(q2)
    T_2_3 = _trans(-0.40718, 0.0, 0.09764) @ _rot_y(-q3)
    T_0_3 = T_0_1 @ T_1_2 @ T_2_3
    
    return T_0_3 @ _trans(-0.05908, 0.0, 0.44528)


def _head_l3_pose_from_joints(
    q_torso: np.ndarray | list[float],
    q_head: np.ndarray | list[float],
) -> np.ndarray:
    """head_l3의 base frame 4x4 pose 반환."""
    q_t = np.asarray(q_torso, dtype=np.float64).ravel()[:3]
    q_h = np.asarray(q_head, dtype=np.float64).ravel()[:3]
    h1, h2, h3 = q_h[0], q_h[1], q_h[2]
    
    T_base_ac = _T_base_arm_center(q_t)
    T_ac_l1 = _trans(-0.0735, -0.0725, 0.014) @ _rot_y(h1)
    T_l1_l2 = _trans(0.0, 0.0725, -0.0035) @ _rot_z(h2)
    T_l2_l3 = _trans(0.0, 0.002, 0.0495) @ _rot_y(-h3)
    
    return T_base_ac @ T_ac_l1 @ T_l1_l2 @ T_l2_l3


def _zed_left_camera_pose_from_joints(
    q_torso: np.ndarray | list[float],
    q_head: np.ndarray | list[float],
) -> np.ndarray:
    """zed_left_camera의 base frame 4x4 pose 반환."""
    T_base_l3 = _head_l3_pose_from_joints(q_torso, q_head)
    T_l3_cam = _trans(0.0365, 0.023, 0.0489) @ _rot_rpy(-1.57079, 0, -1.57079)
    return T_base_l3 @ T_l3_cam


def transform_zed_point_to_base(
    point_in_zed: np.ndarray,
    q_torso: np.ndarray | list[float],
    q_head: np.ndarray | list[float],
) -> np.ndarray:
    """ZED 카메라 좌표계의 점을 베이스 좌표계로 변환."""
    T_base_cam = _zed_left_camera_pose_from_joints(q_torso, q_head)
    point_zed_homo = np.array([point_in_zed[0], point_in_zed[1], point_in_zed[2], 1.0], dtype=np.float64)
    point_base_homo = T_base_cam @ point_zed_homo
    return point_base_homo[:3]


def get_camera_data(robot):
    """Simple function to get head ZED X Mini camera data from robot sensors.

    This demonstrates how easy it is to get camera data using our API.
    """
    return robot.sensors.head_camera.get_obs(
        obs_keys=["left_rgb", "depth"], include_timestamp=True
    )


def print_camera_info(camera_info):
    """Nicely format and print any nested dictionary."""

    def print_dict(d, indent=0):
        """Recursively print dictionary with proper indentation."""
        for key, value in d.items():
            # Create indentation
            spaces = "  " * indent

            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, (list, tuple)):
                print(f"{spaces}{key}: {value}")
            else:
                print(f"{spaces}{key}: {value}")

    if not camera_info:
        print("No camera information available")
        return

    print("\n" + "=" * 50)
    print("HEAD ZED X MINI CAMERA INFORMATION")
    print("=" * 50)
    print_dict(camera_info)
    print("=" * 50)
    print()
    
    
def get_3d_zed_point(center_point, depth):
    fx = 366.21429443359375
    fy = 366.21429443359375
    cx = 497.73809814453125
    cy = 315.53277587890625
    u, v = center_point
    z = depth
    x = ((u-cx)/fx)*z
    y = ((v-cy)/fy)*z
    
    return (x,y,z)
    
    





point_queue = Queue(maxsize=1)
yaw_queue = Queue(maxsize=1)
occlusion_queue = Queue(maxsize=1)

class BoxDetectionNode(Node):
    
    def __init__(self):
        super().__init__('box_detection_node')
        
        self.target_result_pub = self.create_publisher(String, "/perception/target_result", 10)
        self.pick_check_result_pub = self.create_publisher(String, "/perception/pick_check_result", 10)
        
        self.pick_check_request_sub = self.create_subscription(String,"/perception/pick_check_request",self._on_next_action,10)

        self.current_base_point = None
        self.yaw = 0.0
        self.occlusion = None
        self.sequence_id=0
        self.operation_type=""


        self.result_list_point = []
        self.result_list_yaw = []
        
        self._lock = threading.Lock()
        self._running = False
        
        
    # ------------------------------------------------------------------
    # Subscription callback
    # ------------------------------------------------------------------

    def _on_next_action(self, msg: String) -> None:
        raw = msg.data.strip()
        try:
            request_msg = json.loads(raw)
        except json.JSONDecodeError as e:
            return
            
        self.sequence_id=request_msg.get('sequence_id')
        self.operation_type=request_msg.get('operation_type')
        self.get_logger().info(f"Get pick check request, sequence_id : {self.sequence_id}, operation_type : {self.operation_type}")

        if not point_queue.empty():
            point_queue.get_nowait()
        if not yaw_queue.empty():
            yaw_queue.get_nowait()
        if not occlusion_queue.empty():
            occlusion_queue.get_nowait()

        #time.sleep(2)
        
        with self._lock:
            if self._running:
                return
            self._running = True

        try_cnt = 0
        while len(self.result_list_point) < 3:
            time.sleep(0.1)
            if not point_queue.empty():
                self.result_list_point.append(np.array(point_queue.get_nowait()))
                self.result_list_yaw.append(np.array(yaw_queue.get_nowait()))
            try_cnt+=1
            if try_cnt==50:
                break

        if try_cnt==50:
            self.current_base_point=None
        else:
            self._get_median()

        self._publish_detection()
        self.get_logger().info(f"Publish Detection, {self.sequence_id}, {self.operation_type}")
        self.result_list_point = []
        self.result_list_yaw = []

    def _get_median(self):
        point_median = np.median(self.result_list_point, axis=0)
        median_index = np.argmin(np.abs(self.result_list_point - point_median),axis=0)[0]

        self.current_base_point = self.result_list_point[median_index]
        self.yaw = self.result_list_yaw[median_index]


    def _get_delta(self):
        can_pick = True
        delta={"x": 0.0, "y": 0.0, "yaw" : 0.0}

        x_threshold = 0.05
        x_center = 0.68

        limit_yaw_value = 0.8
        limit_xy_value = 0.4

        adjust_yaw_factor = 0.5
        adjust_xy_factor = 0.9

        if "VegaTask2" in self.operation_type or "VegaTask4" in self.operation_type:
            x_center = 0.55
            adjust_xy_factor = 0.7

        y_threshold = 0.05
        yaw_threshold = 0.2

        nav_tolerance = 0.03

        if abs(self.yaw)>yaw_threshold:
            delta["yaw"] = np.float64(self.yaw)*adjust_yaw_factor
            delta["yaw"] = np.clip(delta["yaw"], -limit_yaw_value, limit_yaw_value)
            can_pick=False
            return can_pick, delta

        if abs(self.current_base_point[0] - x_center) > x_threshold:
            if abs(self.current_base_point[0] - x_center)*adjust_xy_factor > nav_tolerance:
                delta["x"] = (self.current_base_point[0] - x_center)*adjust_xy_factor
                delta["x"] = np.clip(delta["x"], -limit_xy_value, limit_xy_value)
                can_pick = False

        if abs(self.current_base_point[1]) > y_threshold:
            if abs(self.current_base_point[1])*adjust_xy_factor > nav_tolerance:
                delta["y"] = self.current_base_point[1]*adjust_xy_factor
                delta["y"] = np.clip(delta["y"], -limit_xy_value, limit_xy_value)
                can_pick = False


        return can_pick, delta
    
    def _publish_detection(self):
        try:        
            # 큐에서 최신 데이터 가져오기
            #if not point_queue.empty():
            #    self.current_base_point = point_queue.get_nowait()

            #if not yaw_queue.empty():
            #    self.yaw = yaw_queue.get_nowait()

            if not occlusion_queue.empty():
                self.occlusion = occlusion_queue.get_nowait()

            check_result=String()
            check_ = {
                "sequence_id": self.sequence_id,
                "operation_type": self.operation_type,
                "can_pick": False,
                "delta": {"x": 0.0, "y": 0.0, "yaw" : 0.0}
            }
            # 포인트가 있으면 발행
            if self.current_base_point is not None:
                can_pick, delta = self._get_delta()
                if "test" in self.operation_type:
                    can_pick = True
                check_["can_pick"] = can_pick
                check_["delta"] = delta
                check_["position"] =  {
                "x": self.current_base_point[0],
                "y": self.current_base_point[1],
                "z": self.current_base_point[2]
                }
                print(delta)
                self.get_logger().info(f"Published base_coord: x={self.current_base_point[0]:.4f}, y={self.current_base_point[1]:.4f}, z={self.current_base_point[2]:.4f}")
                self.current_base_point = None
            elif self.occlusion is not None:
                if self.occlusion>0:
                    y = 0.2
                else:
                    y = -0.2
                check_["delta"]["y"]=y
                print(check_["delta"])
                self.get_logger().info("Detect Occlusion")
                self.occlusion = None
    
            else:
                self.get_logger().info("Detection Failed")

            check_result.data = json.dumps(check_)
            self.pick_check_result_pub.publish(check_result)
            self._running=False
            
                
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")
            self._running=False
        


def send_camera_data(robot, text):
    device = Accelerator().device
    #model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only=True).to(device)
    #processor = AutoProcessor.from_pretrained("./grounding-dino-local")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # ROS2 초기화
    rclpy.init()
    node = BoxDetectionNode()
    
    # ROS2 노드 스핀을 별도 스레드에서 실행
    def run_ros2():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    
    ros2_thread = threading.Thread(target=run_ros2, daemon=True)
    ros2_thread.start()

    stream_names = ["left_rgb","depth"]
    
    text_labels=[[text]]

    def get_position_data(image, text_labels=text_labels):
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.shape[:2]]
        )

        return results[0]

    try:
        while True:
            # Get camera data - simple API call
            camera_data = get_camera_data(robot)
            is_detected = False
            is_occluded = False
            for i, key in enumerate(stream_names):
                if key in camera_data and camera_data[key] is not None:
                    data = camera_data[key]

                    # Extract image, publish timestamp, and receive timestamp
                    if isinstance(data, dict):
                        img = data.get("data")
                        timestamp_ns = data.get("timestamp_ns")
                        receive_time_ns = data.get("receive_time_ns")
                    else:
                        img = data
                        timestamp_ns = None
                        receive_time_ns = None
                    # Skip if no image data
                    if img is None:
                        continue
                        
                    if "rgb" in key:
                        #inf_start = time.time()
                        result = get_position_data(img, text_labels)
                        #print("Inference: " ,time.time()-inf_start)
                        #print(result)
                        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
                            if labels not in text_labels[0]:
                                continue
                            box = [int(round(x, 0)) for x in box.tolist()]
                            u = min(int((box[0]+box[2])/2),img.shape[1])
                            v = min(int((box[1]+box[3])/2),img.shape[0])
                            v_check = int((v+box[3]*3)/4)
                            center_point = (u,v)
                            direction = img.shape[1]//2 - u
                            
                            is_detected=True
                            break
                    
                    if "depth" in key and is_detected:
                        u, v = center_point
                        depth_value = img[v_check,u]
                        depth_left = img[v_check,u-10]
                        depth_right=img[v_check,u+10]

                        min_depth_value = min(depth_value, depth_left, depth_right)
                        max_depth_value = max(depth_value, depth_left, depth_right)
                        if min_depth_value<0.5 or max_depth_value>1.2:
                            is_detected=False
                        if min_depth_value<0.5:
                            is_occluded=True

                        yaw = 0.0
                        if abs(depth_right-depth_left)>0.005:
                            left_point = get_3d_zed_point((u-10,v_check), depth_left)
                            right_point = get_3d_zed_point((u+10,v_check), depth_right)
                            yaw = np.atan2(right_point[2]-left_point[2],right_point[0]-left_point[0])

                        
                            
                        
            if is_detected:
                target_point = get_3d_zed_point(center_point, depth_value)
                
                q_torso = np.array(robot.torso.get_joint_pos(), dtype=np.float64)
                q_head = np.array(robot.head.get_joint_pos(), dtype=np.float64)
                base_point = transform_zed_point_to_base(target_point, q_torso, q_head)
                #print(base_point)
                if not point_queue.empty():
                    point_queue.get()
                if not yaw_queue.empty():
                    yaw_queue.get()
                point_queue.put(base_point)
                yaw_queue.put(yaw)

                is_detected=False

            if is_occluded:
                if not occlusion_queue.empty():
                    occlusion_queue.get()
                occlusion_queue.put(direction)



    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
       


def main(use_rtc: bool = False) -> None:
    """Main function to initialize robot and display head ZED X Mini camera feeds.

    Args:
        fps: Display refresh rate in Hz (default: 30.0)
        use_rtc: Use WebRTC for RGB streams if True (default: False)
    """
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "rtc" if use_rtc else "zenoh"

    with Robot(configs=configs) as robot:
        # Wait for camera to become active
        print("Waiting for camera streams to become active...")
        if robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Camera streams active!")
        else:
            print("Warning: Some camera streams may not be active")

        align_head_to_forward(robot)

        # Print camera information nicely
        #camera_info = robot.sensors.head_camera.get_camera_info()
        #print_camera_info(camera_info)

        # Start live camera visualization
        #visualize_camera_data(robot, fps)
        
        send_camera_data(robot, "green box")
        #detect_thread = threading.Thread(target=send_camera_data, kwargs={'robot':robot, 'text':"green box"},daemon=True)
        #detect_thread.start()
        #detect_thread.join()


if __name__ == "__main__":
    tyro.cli(main)
