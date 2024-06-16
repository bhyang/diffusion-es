vehicle = self.get_vehicle(17)
if vehicle.distance_to(self.ego_vehicle) < 20.0:
    done = self.yield_to_vehicle(vehicle)
    while not done():
        yield
    self.stop_yielding()