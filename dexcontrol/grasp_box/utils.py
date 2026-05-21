import sys
import termios
import tty
from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


def wait_for_space():
    """Wait for user to press space bar before continuing."""
    logger.info("Press SPACE to continue to next waypoint (or 'q' to quit)...")
    
    # Save terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        # Set terminal to raw mode
        tty.setraw(fd)
        
        while True:
            char = sys.stdin.read(1)
            if char == ' ':  # Space bar
                logger.info("Continuing to next waypoint...")
                break
            elif char.lower() == 'q':  # Quit
                logger.warning("Quit requested by user")
                raise KeyboardInterrupt
            elif char == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def parse_trajectory_file(filepath: str) -> list[dict[str, Any]]:
    """Parse the trajectory file.

    Supported step types (detected from keyword on first line):

    TORSO step — move torso joints directly (values in degrees):
        q1, q2, q3, TORSO
        duration

    JOINT step — send arm joints directly without IK (values in radians):
        j1, j2, j3, j4, j5, j6, j7, JOINT   # left arm
        j1, j2, j3, j4, j5, j6, j7, JOINT   # right arm
        duration

    IK step — solve IK for Cartesian end-effector targets (arm_center frame):
        x, y, z, roll, pitch, yaw[, BOX|REL]  # left arm
        x, y, z, roll, pitch, yaw[, BOX|REL]  # right arm
        duration

    All step types accept an optional pause_after (seconds) as a final line.

    Args:
        filepath: Path to the trajectory file.

    Returns:
        List of waypoint dicts, each with a 'type' key ('torso'|'joint'|'ik').
    """
    waypoints = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.strip() for line in lines
             if line.strip() and not line.strip().startswith('#')]

    def _try_pause(idx: int) -> tuple[float, int]:
        """Try to read an optional pause_after at lines[idx]; return (value, new_i)."""
        if idx < len(lines) and ',' not in lines[idx]:
            try:
                return float(lines[idx].strip()), idx + 1
            except ValueError:
                pass
        return 0.0, idx

    i = 0
    while i < len(lines):
        try:
            first_parts = [x.strip() for x in lines[i].split(',')]

            # ── TORSO step: "q1, q2, q3, TORSO" ──────────────────────────────
            if len(first_parts) == 4 and first_parts[3].upper() == 'TORSO':
                joint_angles_deg = [float(x) for x in first_parts[:3]]
                duration = float(lines[i + 1].strip())
                pause_after, i = _try_pause(i + 2)
                waypoints.append({
                    'type': 'torso',
                    'joint_angles_deg': joint_angles_deg,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── JOINT step: "j1..j7, JOINT" for both arms ─────────────────────
            if len(first_parts) == 8 and first_parts[7].upper() == 'JOINT':
                left_joints = np.array([np.nan if x == '-' else float(x) for x in first_parts[:7]])
                right_parts = [x.strip() for x in lines[i + 1].split(',')]
                right_joints = np.array([np.nan if x == '-' else float(x) for x in right_parts[:7]])
                duration = float(lines[i + 2].strip())
                pause_after, i = _try_pause(i + 3)
                waypoints.append({
                    'type': 'joint',
                    'left_arm_joints': left_joints,
                    'right_arm_joints': right_joints,
                    'duration': duration,
                    'pause_after': pause_after,
                })
                continue

            # ── GRAB step: "GRAB, <step_size_m>, <max_steps>" ─────────────────
            # Moves both arms inward along Y (left: −Y, right: +Y) by step_size
            # each iteration until either hand reports force ≥ threshold or the
            # hand separation drops below the configured minimum.
            if len(first_parts) >= 1 and first_parts[0].upper() == 'GRAB':
                step_size = float(first_parts[1]) if len(first_parts) > 1 else 0.005
                max_steps = int(first_parts[2]) if len(first_parts) > 2 else 100
                pause_after, i = _try_pause(i + 1)
                waypoints.append({
                    'type': 'grab',
                    'step_size': step_size,
                    'max_steps': max_steps,
                    'pause_after': pause_after,
                })
                continue

            # ── IK step: "x, y, z, roll, pitch, yaw[, BOX|REL]" ──────────────
            # A bare '-' token means "keep current EE value for this field".
            def _parse_val(s: str) -> float:
                return np.nan if s.strip() == '-' else float(s)

            left_use_box = False
            left_use_rel = False
            if len(first_parts) == 7:
                flag = first_parts[6].upper()
                left_use_box = flag == 'BOX'
                left_use_rel = flag == 'REL'
                first_parts = first_parts[:6]
            left_data = [_parse_val(x) for x in first_parts]
            if len(left_data) != 6:
                raise ValueError(f"Line {i+1}: Expected 6 values for left pose")

            right_parts = [x.strip() for x in lines[i + 1].split(',')]
            right_use_box = False
            right_use_rel = False
            if len(right_parts) == 7:
                flag = right_parts[6].upper()
                right_use_box = flag == 'BOX'
                right_use_rel = flag == 'REL'
                right_parts = right_parts[:6]
            right_data = [_parse_val(x) for x in right_parts]
            if len(right_data) != 6:
                raise ValueError(f"Line {i+2}: Expected 6 values for right pose")

            duration = float(lines[i + 2].strip())
            pause_after, i = _try_pause(i + 3)
            left_arr  = np.array(left_data,  dtype=float)
            right_arr = np.array(right_data, dtype=float)
            waypoints.append({
                'type': 'ik',
                'left_pose': {
                    'position': left_arr[:3],
                    'rpy': left_arr[3:6],
                    'use_box': left_use_box,
                    'use_rel': left_use_rel,
                    'use_cur': bool(np.any(np.isnan(left_arr))),
                },
                'right_pose': {
                    'position': right_arr[:3],
                    'rpy': right_arr[3:6],
                    'use_box': right_use_box,
                    'use_rel': right_use_rel,
                    'use_cur': bool(np.any(np.isnan(right_arr))),
                },
                'duration': duration,
                'pause_after': pause_after,
            })

        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing trajectory at line {i+1}: {e}")
            break

    logger.info(f"Parsed {len(waypoints)} waypoints from trajectory file")
    return waypoints


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw angles to rotation matrix (extrinsic XYZ convention).
    
    Args:
        roll: Rotation around X-axis (radians).
        pitch: Rotation around Y-axis (radians).
        yaw: Rotation around Z-axis (radians).
        
    Returns:
      3x3 rotation matrix.
    """
    # Extrinsic XYZ rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rot.as_matrix()


def align_head_to_forward(bot, wait_time: float = 6.0) -> None:
    """Adjust head pitch so the robot looks forward in the world frame.

    Keeps torso_pitch_deg + (-head_pitch_deg) ≈ 90 deg, meaning the
    head faces the horizon regardless of how much the torso is tilted.

    Args:
        bot: Connected Robot instance.
        wait_time: Maximum time (seconds) to wait for the head to reach
                   the target position.
    """
    forward_sum_deg = 70.0
    torso_pitch_deg = float(np.rad2deg(bot.torso.pitch_angle))
    current_head_pos = np.asarray(bot.head._get_state()["pos"], dtype=float)
    print(current_head_pos)
    target_head_pos = np.zeros_like(current_head_pos)
    target_head_pos[0] = np.deg2rad(torso_pitch_deg - forward_sum_deg)

    head_error = target_head_pos - current_head_pos
    head_kp = 0.6
    head_min_vel = 0.02
    head_max_vel = 1.0
    head_joint_vel = np.clip(np.abs(head_error) * head_kp, head_min_vel, head_max_vel)

    bot.head.set_joint_pos_vel(
        joint_pos=target_head_pos,
        joint_vel=head_joint_vel,
        wait_time=wait_time,
        exit_on_reach=True,
    )

