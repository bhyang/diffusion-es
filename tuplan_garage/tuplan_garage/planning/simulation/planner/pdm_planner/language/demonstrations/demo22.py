current_speed = self.ego_vehicle.speed
self.set_ego_speed_limit(current_speed * 0.5)
done = self.follow_lane(self.right_lane)
while not done():
    yield