vehicle = self.get_vehicle(1)
if not vehicle.is_stopped():
    done = self.yield_to_vehicle(vehicle)
    while not done():
        yield
    self.stop_yielding()