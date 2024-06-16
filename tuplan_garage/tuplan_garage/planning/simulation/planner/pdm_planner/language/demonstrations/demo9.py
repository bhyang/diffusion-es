vehicle = self.get_vehicle(2)
if vehicle.speed < self.current_lane.speed_limit:
    done = self.follow_lane(self.right_lane)
    while not done():
        yield