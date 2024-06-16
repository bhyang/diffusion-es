def make_plan():
    current_lane, left_lanes, right_lanes = get_adjacent_lane_centers()
    if len(left_lanes) == 0:
        yield 'done'
        return
    left_lane = left_lanes[0]
    offset = np.array([50,0])
    goal = transform_to_pose_frame(offset, left_lane)
    set_static_goal(goal)
    while not goal_reached():
        yield 'running'
    yield 'done'
    return