"""Battery pick — Module 1 (Perception) and Module 3 (Suction Grasp).

Public API::

    from battery_pick import BatteryDetector, BatteryObservation
    from battery_pick import SuctionGrasper, GraspResult

    with Robot() as bot:
        detector = BatteryDetector(bot)
        grasper  = SuctionGrasper(bot)

        batteries = detector.detect()       # Module 1
        result    = grasper.grasp(batteries[0])  # Module 3
        if result.success:
            grasper.release()
        grasper.close()
"""

from battery_pick.perception import BatteryDetector, BatteryObservation
from battery_pick.suction_grasp import GraspResult, SuctionGrasper

__all__ = [
    "BatteryDetector",
    "BatteryObservation",
    "SuctionGrasper",
    "GraspResult",
]
