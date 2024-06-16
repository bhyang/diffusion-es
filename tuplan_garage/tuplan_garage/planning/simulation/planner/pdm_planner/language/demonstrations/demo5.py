vehicle = self.get_vehicle(8)
if vehicle.speed > 5.0:
    done = self.yield_to_vehicle(vehicle)
    while not done():
        yield
    self.stop_yielding()