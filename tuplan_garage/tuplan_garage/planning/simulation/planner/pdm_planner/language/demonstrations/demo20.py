lane = self.get_lane(1)
done = self.follow_lane(lane)
while not done():
    yield