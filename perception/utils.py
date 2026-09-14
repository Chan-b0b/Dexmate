import numpy as np


def set_head_pitch(bot, wait_time: float = 5.0, angle: float = 30.0,
                   tol_deg: float | None = None) -> None:
    """Adjust head pitch so the robot looks forward in the world frame.

    Keeps torso_pitch_deg + (-head_pitch_deg) ≈ 90 deg, meaning the
    head faces the horizon regardless of how much the torso is tilted.

    Args:
        bot: Connected Robot instance.
        wait_time: Maximum time (seconds) to wait for the head to reach
                   the target position.
        angle: Desired angle for the head to look forward.
        tol_deg: When given, RETURN IMMEDIATELY if the head is already within
                 this many degrees of the target, and exit the move as soon as
                 it arrives instead of always blocking for ``wait_time``.
                 Default None keeps the original behaviour for every caller
                 that has not opted in: a fixed ``wait_time`` block, which also
                 acts as a settle pause after whatever else just moved.
                 0911: the box pick spent a flat 5 s here on a head that was
                 already at 24 deg from the previous task.
    """
    forward_sum_deg = angle
    torso_pitch_deg = float(np.rad2deg(bot.torso.pitch_angle))
    current_head_pos = np.asarray(bot.head.get_state()["pos"], dtype=float)
    
    target_head_pos = np.zeros_like(current_head_pos)
    target_head_pos[0] = np.deg2rad(torso_pitch_deg - forward_sum_deg)

    head_error = target_head_pos - current_head_pos
    if tol_deg is not None and float(np.max(np.abs(np.rad2deg(head_error)))) <= float(tol_deg):
        return
    head_kp = 0.6
    head_min_vel = 0.02
    head_max_vel = 1.0
    head_joint_vel = np.clip(np.abs(head_error) * head_kp, head_min_vel, head_max_vel)

    bot.head.set_joint_pos_vel(
        joint_pos=target_head_pos,
        joint_vel=head_joint_vel,
        wait_time=wait_time,
        exit_on_reach=tol_deg is not None,
    )
