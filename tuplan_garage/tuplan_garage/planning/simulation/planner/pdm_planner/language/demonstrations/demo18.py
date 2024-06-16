vehicle = self.get_vehicle(2)
if vehicle.is_ahead_of(self.ego_vehicle, 10.0):
    self.adjust_constant_velocity_prediction(vehicle, 0.5)