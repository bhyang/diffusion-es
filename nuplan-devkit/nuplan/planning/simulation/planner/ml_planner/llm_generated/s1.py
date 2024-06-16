# In 10 seconds, change lanes to the right
def make_plan():
    for _ in range(seconds_to_steps(10)):
        yield 'running'
    current_lane, left_lanes, right_lanes = get_adjacent_lane_centers()
    right_lane = right_lanes[0]
    offset = np.array([10,0])
    goal = transform_to_pose_frame(offset, right_lane)
    set_static_goal(goal)
    while not goal_reached():
        yield 'running'
    yield 'done'