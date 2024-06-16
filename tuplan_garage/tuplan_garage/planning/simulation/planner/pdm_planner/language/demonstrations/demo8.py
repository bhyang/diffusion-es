vehicles_in_left_lane = self.left_lane.get_vehicles()
if any([vehicle.speed > 5.0 for vehicle in vehicles_in_left_lane]):
    done = self.follow_lane(self.current_lane)
    while not done():
        yield