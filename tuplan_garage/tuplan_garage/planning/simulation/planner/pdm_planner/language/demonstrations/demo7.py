def vehicle_ahead_of_us():
    vehicle = self.get_vehicle(3)
    return vehicle.is_ahead_of(self.ego_vehicle)
while vehicle_ahead_of_us():
    self.follow_lane(self.current_lane)
    yield
vehicle = self.get_vehicle(3)
their_lane = vehicle.get_closest_lane(self.lane_graph)
done = self.follow_lane(their_lane)
while not done():
    yield