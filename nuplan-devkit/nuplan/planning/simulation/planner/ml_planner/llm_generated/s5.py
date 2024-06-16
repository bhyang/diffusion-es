def make_plan():
    current_lane, left_lanes, right_lanes = get_adjacent_lane_centers()
    if len(right_lanes) > 0:
        right_lane = right_lanes[0]
    else:
        right_lane = current_lane
    offset = np.array([10,0])
    goal = transform_to_pose_frame(offset, right_lane)
    set_static_goal(goal)
    while not goal_reached():
        yield 'running'
    yield 'done'