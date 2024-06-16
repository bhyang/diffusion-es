vehicle = self.get_vehicle(3)
done = self.yield_to_vehicle(vehicle)
while not done():
    yield
self.stop_yielding()