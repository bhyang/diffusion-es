done = self.follow_lane(self.right_lane)
while not done():
    yield
done = self.follow_lane(self.left_lane)
while not done():
    yield