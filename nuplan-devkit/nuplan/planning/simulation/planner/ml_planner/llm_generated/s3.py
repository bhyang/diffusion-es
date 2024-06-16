def make_plan():
    goal = np.array([50,0])
    set_static_goal(goal)
    while not goal_reached():
        yield 'running'
    yield 'done'