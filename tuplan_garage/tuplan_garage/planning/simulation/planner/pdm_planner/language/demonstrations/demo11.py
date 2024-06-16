done = self.follow_lane(self.right_lane)
while not done():
    yield
def vehicle_ahead_of_us():
    vehicle = self.get_vehicle(1)
    return vehicle.is_ahead_of(self.ego_vehicle)
while not vehicle_ahead_of_us():
    yield
done = self.yield_to_vehicle(self.get_vehicle(1))
while not done():
    yield
self.stop_yielding()