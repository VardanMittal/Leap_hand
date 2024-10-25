import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/vardan/Leap_hand/ros2_ws/install/Leap_hand'
