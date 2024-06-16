def vehicle_is_ever_stopped():
    vehicle = self.get_vehicle(20)
    return vehicle.is_stopped()
while not vehicle_is_ever_stopped():
    yield
vehicle = self.get_vehicle(20)
done = self.yield_to_vehicle(vehicle)
while not done():
    yield
self.stop_yielding()