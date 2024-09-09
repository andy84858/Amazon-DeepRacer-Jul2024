import math

# =====================# HELPER FUNCTION # =====================#


def dist_2_points(x1, x2, y1, y2):
    """
    計算兩點之間的歐幾里得距離
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def calculate_alignment_reward(params):
    """
    計算車輛與目標方向的對齊獎勵。

    參數：
    params: 包含車輛當前狀態的字典。

    返回值：
    對齊獎勵值。
    """
    # 讀取車輛當前位置座標
    x = params['x']
    y = params['y']

    # 讀取賽道的waypoints
    waypoints = params['waypoints']

    # 找到接下來的3個waypoints的索引
    next_points = find_next_three_waypoints(params)

    # 獲取第3個waypoint（目標點）的座標
    x_forward = waypoints[next_points[2]][0]
    y_forward = waypoints[next_points[2]][1]

    # 讀取車輛當前的朝向
    heading = params['heading']

    # 計算從車輛當前位置到目標點的最佳方向（以度數表示）
    optimal_heading = math.degrees(math.atan2(y_forward - y, x_forward - x))

    # 計算車輛朝向與最佳方向之間的差異
    heading_diff = abs(optimal_heading - heading)

    # 確保角度差在0到180度之間
    if heading_diff > 180:
        heading_diff = 360 - heading_diff

    # 計算對齊獎勵，使用cos函數使得對齊差異越小，獎勵越高
    reward_alignment = math.cos(math.radians(heading_diff))

    return reward_alignment


def find_next_three_waypoints(params):
    """
    找到接下來的3個waypoints。

    參數：
    params: 包含賽車當前狀態的字典。

    返回值：
    接下來的兩個waypoints的索引列表。
    """
    # 讀取賽道的waypoints
    waypoints = params['waypoints']

    # 獲取最近的waypoint索引
    closest_waypoint_index = params['closest_waypoints'][1]

    # 計算接下來的三個waypoints的索引
    next_points = list(range(closest_waypoint_index, closest_waypoint_index + 3))

    # 確保索引在waypoints範圍內（處理索引超過範圍的情況）
    for i in range(len(next_points)):
        if next_points[i] >= len(waypoints):
            next_points[i] -= len(waypoints)

    return next_points

class Reward:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def reward_function(self, params):
        """
        根據車輛狀態給予獎勵。

        參數：
        params: 賽車當前狀態的參數字典。

        返回值：
        獎勵值。
        """

        # 讀取賽車狀態參數
        is_offtrack = params['is_offtrack']
        x = params['x']
        y = params['y']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        track_width = params['track_width']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']

        # 預期最佳每個sector的時間
        optimal_sector_1_time = 4.38
        optimal_sector_2_time = 6.55
        optimal_sector_3_time = 5.35

        # 每個Sector每秒的步數
        sector_1_steps_per_second = 12.79
        sector_2_steps_per_second = 8.54
        sector_3_steps_per_second = 10.65

        # 預期最佳每個sector的時間
        best_step1 = optimal_sector_1_time * sector_1_steps_per_second
        best_step2 = optimal_sector_2_time * sector_2_steps_per_second
        best_step3 = optimal_sector_3_time * sector_3_steps_per_second

        # 定義初始獎勵
        reward = 1.0

        # =============== 1. 獎勵在預估秒數內完成sector ================ #
        sector = (steps - 1) // (len(waypoints) // 3) + 1
        if sector == 1:
            if steps < best_step1:
                reward += 50
        elif sector == 2:
            if steps < best_step2:
                reward += 50
        elif sector == 3:
            if steps < best_step3:
                reward += 50
        # =========================================================== #

        # ====== 2. 鼓勵找到最遠的waypoints，並朝該點前進，不斷更新 ====== #

        # 找到接下來的三個 waypoints
        next_points = find_next_three_waypoints(params)
        next_waypoint = waypoints[next_points[2]]

        # 計算車輛當前位置到下一個 waypoint 的距離
        current_position = (x, y)
        next_waypoint_distance = dist_2_points(current_position[0], next_waypoint[0], current_position[1], next_waypoint[1])

        # 計算車輛與目標方向的對齊獎勵
        reward += calculate_alignment_reward(params) * speed

        # =========================================================== #

        # ============ 3. 指定直線、彎路路段waypoint index ============= #

        # 指定的直線路段的waypoint索引
        straight_section1 = list(range(0, 21))        # 靠右
        straight_section2 = list(range(32, 36))       # cross
        straight_section3 = list(range(46, 53))       # 靠右
        straight_section4 = list(range(95, 100))      # 靠右
        straight_section5 = list(range(107, 123))     # 靠右
        straight_section6 = list(range(130, 150))     # 靠右 

        # 指定的急彎路段的waypoint索引
        hairpin_section1 = list(range(72, 78))
        hairpin_section2 = list(range(89, 92))
        hairpin_section3 = list(range(158, 163))
        
        # 確認當前車輛的位置索引
        current_index = closest_waypoints[0]

        # 判斷是否在指定的直線路段
        if current_index in straight_section1 or current_index in straight_section6:
            # 獎勵靠右
            if speed > 3 and abs(steering_angle) < 3:
                reward += speed * 3
                if not is_left_of_center:
                    reward += 2
                
        elif (current_index in straight_section2 or current_index in straight_section3 or
             current_index in straight_section4 or current_index in straight_section5):
            if speed > 3 and abs(steering_angle) < 3:
                    reward += speed * 3

        # 判斷是否在急彎路段
        if current_index in hairpin_section1 or current_index in hairpin_section3:
            # 左彎
            if steering_angle > 25:
                reward += 5
        elif current_index in hairpin_section2:
            # 右彎
            if steering_angle < -25:
                reward += 5

        # =========================================================== #


        # 錯誤方向（spin）大大扣分
        if abs(steering_angle) > 30:
            reward -= 5

        # 超出賽道大大扣分
        if is_offtrack:
            reward -= 20

        return float(reward)


reward_object = Reward(verbose=True)

def reward_function(params):
    return reward_object.reward_function(params)