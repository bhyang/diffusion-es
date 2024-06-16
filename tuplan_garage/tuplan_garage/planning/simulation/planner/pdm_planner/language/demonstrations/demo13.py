done = self.yield_to_vehicle(self.get_vehicle(4))
def vehicle_ahead_of_other_vehicle():
    vehicle = self.get_vehicle(2)
    other_vehicle = self.get_vehicle(4)
    return vehicle.is_ahead_of(other_vehicle)
while not done() and not vehicle_ahead_of_other_vehicle():
    yield
self.stop_yielding()