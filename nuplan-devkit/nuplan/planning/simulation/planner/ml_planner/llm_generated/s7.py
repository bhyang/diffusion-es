# Change to the rightmost lane
def make_plan():
    current_lane, left_lanes, right_lanes = get_adjacent_lane_centers()
    rightmost_lane = right_lanes[-1]
    offset = np.array([10,0])
    goal = transform_to_pose_frame(offset, rightmost_lane)
    set_static_goal(goal)
    while not goal_reached():
        yield 'running'
    yield 'done'