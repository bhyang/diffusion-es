current_speed = self.ego_vehicle.speed
self.set_ego_speed_limit(current_speed * 0.2)
done = self.follow_lane(self.right_lane)
while not done():
    yield
def vehicle_ahead_of_us():
    vehicle = self.get_vehicle(2)
    return vehicle.is_ahead_of(self.ego_vehicle)
while not vehicle_ahead_of_us():
    yield
current_speed = self.ego_vehicle.speed
self.unset_ego_speed_limit()