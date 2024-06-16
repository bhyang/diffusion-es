speed_limit = self.current_lane.speed_limit
def speed_exceeds_limit():
    vehicle = self.get_vehicle(2)
    return vehicle.speed < speed_limit
while speed_exceeds_limit():
    yield
done = self.follow_lane(self.right_lane)
while not done():
    yield