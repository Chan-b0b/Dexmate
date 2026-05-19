import numpy as np


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
