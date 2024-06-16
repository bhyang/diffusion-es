done = self.follow_lane(self.current_lane)
def speed_under_threshold():
    speed = self.ego_vehicle.speed
    return speed < 2.0
while not done() and speed_under_threshold():
    yield
self.stop_following()