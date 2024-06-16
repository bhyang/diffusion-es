def vehicle_ahead_of_us():
    vehicle = self.get_vehicle(2)
    return vehicle.is_ahead_of(self.ego_vehicle, 10.0)
while not vehicle_ahead_of_us():
    yield
self.set_velocity_ratio(self.get_vehicle(2), 0.5)