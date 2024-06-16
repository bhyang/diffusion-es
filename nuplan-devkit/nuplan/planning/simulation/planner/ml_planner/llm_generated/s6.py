def make_plan():
    headings, distances = get_heading_and_distance_to_agents()
    agent_ids = np.arange(headings.shape[0])
    in_front = np.abs(headings) < np.pi / 3
    closest_id = agent_ids[in_front][np.argmin(headings[in_front])]
    agent_poses = get_agent_poses()
    goal_pos = agent_poses[closest_id,:2]
    set_static_goal(goal_pos)
    while not goal_reached():
        yield 'running'
    yield 'done'