situation_bundle — Zenoh telemetry → ROS 2 + Task Manager → /situation/next_action
===================================================================================

Contents
--------
  bridge/zenoh_robot_state_to_ros2.py   Zenoh → ROS 2 (optional)
  manipulation_bridge/manipulation_manager.py
    Sub /manipulation/command → Pub /manipulation/state, /situation/next_action
    Initialize: Stand → perception → /manipulation/state DONE (payload: can_pick).
      If can_pick is false, next same Initialize skips Stand and only repeats pick_check request.
      Optional JSON: "skip_stand": true|false to force perception-only or full Stand+perception.
  situation_stack/                      sources ROS, runs the above

Vega tasks (operation_type → same id on /situation/next_action)
---------------------------------------------------------------
  G1Task1-1
  G1Task2-1
  VegaTask1-040-Finish   sequential workflow: 041-Place -> 042-Sit -> /manipulation/state DONE
  VegaTask2-040-Finish   sequential workflow: 041-Place -> 042-Sit -> /manipulation/state DONE
  VegaTask1-010-Initialize  workflow: 011-Stand -> 012-Perception -> /manipulation/state DONE (can_pick in JSON)
  VegaTask2-010-Initialize  workflow: 011-Stand -> 012-Perception -> /manipulation/state DONE (can_pick in JSON)
  VegaTask1-020-Manipulation  workflow: 021-Grab -> /manipulation/state DONE
  VegaTask2-020-Manipulation  workflow: 021-Grab -> /manipulation/state DONE
  VegaTask3-020-Initialize  workflow: 021-Stand -> 022-Perception -> /manipulation/state DONE (can_pick in JSON)
  VegaTask3-030-Manipulation  workflow: 031-Grab -> /manipulation/state DONE
  VegaTask4-020-Finish   sequential workflow: 021-Place -> 022-Sit -> /manipulation/state DONE

  You may send operation_type as a Vega workflow or step id (see list above), or G1 alias:
  g1_task_1_1 → G1Task1-1, g1_task_2_1 → G1Task2-1.

  Example:
    ros2 topic pub /manipulation/command std_msgs/String \
      '{data: "{\"sequence_id\": 1, \"operation_type\": \"VegaTask1-020-Manipulation\"}"}'

Prerequisites
-------------
  - Python 3.10+
  - pip: dexcontrol, dexcomm when using the Zenoh bridge (+ numpy, etc.)
  - ROS 2 Humble (rclpy, std_msgs, …)
  - ZENOH_CONFIG if your Zenoh deployment needs it
  - ROBOT_PREFIX to enable the bridge (optional if you only use manipulation)

Run
---
  chmod +x situation_stack/run.sh run.sh
  export ZENOH_CONFIG=...    # if needed
  export ROBOT_PREFIX=dm/... # for bridge; omit if SITUATION_SKIP_ZENOH_ROS_BRIDGE=1
  ./run.sh


Perception result JSON
----------------------
  Expect JSON on SITUATION_PERCEPTION_RESULT_TOPIC with:
    sequence_id, operation_type, can_pick, offset(optional)
  Example:
    {"sequence_id": 1, "operation_type": "VegaTask1-012-Perception", "can_pick": true,
     "offset": {"x": 0.01, "y": -0.02, "z": 0.0}}

Optional env
------------
  SITUATION_ROS_SETUP       default /opt/ros/humble/setup.bash
  SITUATION_ROS_NAMESPACE   default /vega_1p (bridge ROS namespace)
  DEXCONTROL_PYTHONPATH     prepend if you use dexcontrol from source src/
  SITUATION_SKIP_ZENOH_ROS_BRIDGE=1   do not start the bridge
  SITUATION_SKIP_MANIPULATION_BRIDGE=1   do not start manipulation_manager
  SITUATION_PERCEPTION_REQUEST_TOPIC   default /perception/pick_check_request
  SITUATION_PERCEPTION_RESULT_TOPIC    default /perception/pick_check_result

Copy minimal tree to another machine
--------------------------------------
  ./copy_minimal_runtime.sh [DEST_DIR]
