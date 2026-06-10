from dexcontrol.robot import Robot
from case_battery_demo.robotiq import RobotiqGripper

with Robot() as bot:
    g = RobotiqGripper(bot)
    g.open()
