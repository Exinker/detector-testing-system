import os


# --------        dirs        --------
DIRECTORY = os.path.join('.', 'data')  # os.path.join('.', 'data') or r'\\fileserver\Users\vaschenko\test42'


# --------        device        --------
DEVICE_IP = '10.116.220.2'
DEVICE_UNITS = 'percent'  # 'digit' or 'percent'


# --------        experiment        --------
CHECK_EXPOSURE_FLAG = True
EXPOSURE_MIN = 2
EXPOSURE_MAX = 1000

CHECK_TOTAL_FLAG = False

CHECK_SOURCE_FLAG = True
CHECK_SOURCE_TAU = 5
CHECK_SOURCE_N_FRAMES = 100
