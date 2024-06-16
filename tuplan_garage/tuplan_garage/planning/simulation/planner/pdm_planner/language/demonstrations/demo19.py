lane = self.get_lane(12)
done = self.follow_lane(lane)
while not done():
    yield