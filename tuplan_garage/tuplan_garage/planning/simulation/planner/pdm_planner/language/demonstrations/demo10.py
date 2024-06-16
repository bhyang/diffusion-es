def ahead_of_vehicle():
    vehicle = self.get_vehicle(3)
    return self.ego_vehicle.is_ahead_of(vehicle)
while ahead_of_vehicle():
    self.follow_lane(self.current_lane)
    yield
vehicle = self.get_vehicle(3)
their_lane = vehicle.get_closest_lane(self.lane_graph)
done = self.follow_lane(their_lane)
while not done():
    yield