"""Speed-aware reward for re:Invent 2018 counterclockwise time trials.

Default targets: straight 3.0, gentle corner 2.0, sharp corner 1.5 m/s.
The fast profile requires an action space that supports the
previous candidate's 1.2-3.0 m/s actions. Simulator evaluation is still required.
This is a conservative center/inside reference, not an optimized racing line.
"""

import math

# Action speeds: 0 degrees: 2.0/2.5/3.0; +/-15: 1.6/2.0/2.4; +/-30: 1.2/1.5/1.8 m/s.
# Use "logged" instead if the action space is still limited to 0.5/1.0 m/s.
SPEED_PROFILE = "fast"
SPEED_TARGETS = {
    "logged": (1.0, 1.0, 0.5),
    "fast": (3.0, 2.0, 1.5),
}
MIN_REWARD = 1e-3


def _angle_difference(target, current):
    return (target - current + 180.0) % 360.0 - 180.0


def _bearing(start, end):
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _point_ahead(waypoints, start_index, distance):
    """Interpolate by physical distance so waypoint density does not set preview."""
    point = waypoints[start_index]
    count = len(waypoints)
    for step in range(1, count + 1):
        next_point = waypoints[(start_index + step) % count]
        length = math.hypot(next_point[0] - point[0], next_point[1] - point[1])
        if length > 1e-9:
            if distance <= length:
                fraction = distance / length
                return (
                    point[0] + fraction * (next_point[0] - point[0]),
                    point[1] + fraction * (next_point[1] - point[1]),
                )
            distance -= length
        point = next_point
    return point


def reward_function(params):
    if (params.get("is_offtrack", False)
            or params.get("is_crashed", False)
            or not params["all_wheels_on_track"]):
        return float(MIN_REWARD)

    width = float(params["track_width"])
    waypoints = params["waypoints"]
    if width <= 0.0 or len(waypoints) < 3:
        return float(MIN_REWARD)

    # This file is specifically for increasing-waypoint (CCW) travel.
    if params.get("is_reversed", False):
        return float(MIN_REWARD)

    previous_index, next_index = params["closest_waypoints"]
    track_heading = _bearing(waypoints[previous_index], waypoints[next_index])
    heading_error = abs(_angle_difference(track_heading, params["heading"]))
    if heading_error >= 60.0:
        return float(MIN_REWARD)

    distance_ratio = min(1.0, abs(params["distance_from_center"]) / (0.5 * width))
    signed_position = distance_ratio if params["is_left_of_center"] else -distance_ratio
    speed = max(0.0, float(params["speed"]))
    steering = float(params["steering_angle"])

    # Inspect several future segments, including bends before an S-curve cancels.
    preview = max(0.8 * width, min(1.8 * width, 0.6 * speed))
    points = [_point_ahead(waypoints, next_index, preview * step / 3.0)
              for step in range(4)]
    turns = [_angle_difference(_bearing(points[i], points[i + 1]), track_heading)
             for i in range(3)]
    curvature = max(abs(turn) for turn in turns)
    near_turn = turns[0]

    # Stay near the center; shift at most 7.5% of full width toward the inside.
    # Only the nearest preview segment affects lateral positioning.
    target_position = max(-0.15, min(0.15, near_turn / 100.0))
    lateral_error = signed_position - target_position
    line_factor = math.exp(-2.5 * lateral_error * lateral_error)

    # Point back toward the reference line; allow recovery steering on straights.
    correction = math.degrees(math.atan2(-lateral_error * width * 0.5,
                                        max(0.5 * width, 0.3)))
    desired_heading = track_heading + max(-25.0, min(25.0, correction))
    direction_error = _angle_difference(desired_heading, params["heading"])
    heading_factor = math.exp(-((direction_error / 25.0) ** 2))

    straight_speed, gentle_speed, sharp_speed = SPEED_TARGETS[SPEED_PROFILE]
    if curvature < 8.0:
        target_speed = straight_speed
    elif curvature < 22.0:
        target_speed = gentle_speed
    else:
        target_speed = sharp_speed

    # Large steering, poor alignment or an edge approach must never earn top speed.
    if abs(steering) >= 25.0 or heading_error > 25.0 or distance_ratio > 0.65:
        target_speed = min(target_speed, sharp_speed)
    elif abs(steering) >= 12.0 or heading_error > 12.0:
        target_speed = min(target_speed, gentle_speed)

    relative_speed_error = (speed - target_speed) / max(target_speed, 0.1)
    # Overspeed is more costly than cautious underspeed.
    speed_factor = math.exp(-(6.0 if speed > target_speed else 2.0)
                            * relative_speed_error ** 2)

    steering_factor = 1.0
    if curvature < 8.0 and abs(direction_error) < 8.0 and abs(lateral_error) < 0.2:
        # Once aligned and centered, prefer the zero-steering action.
        steering_factor = math.exp(-((steering / 12.0) ** 2))
    else:
        # Correct heading first; when aligned, follow the nearest bend.
        desired_turn = direction_error if abs(direction_error) > 8.0 else near_turn
        if abs(desired_turn) > 5.0 and steering * desired_turn < 0.0:
            steering_factor = 0.35

    # Apply edge penalties before crossing the boundary, regardless of speed.
    edge_factor = max(0.05, 1.0 - max(0.0, distance_ratio - 0.45) / 0.55)
    if abs(near_turn) > 8.0 and signed_position * near_turn < 0.0:
        edge_factor *= max(0.25, 1.0 - 0.75 * distance_ratio)

    safety_factor = line_factor * heading_factor * edge_factor
    # Speed scaling discourages creeping solely to collect more per-step reward.
    pace_factor = min(1.0, speed / max(straight_speed, 0.1))
    reward = (safety_factor * (0.1 + 0.9 * pace_factor)
              * (0.15 + 0.85 * speed_factor)
              * (0.2 + 0.8 * steering_factor))

    if params.get("progress", 0.0) >= 100.0:
        reward += 5.0

    return float(max(MIN_REWARD, reward))

