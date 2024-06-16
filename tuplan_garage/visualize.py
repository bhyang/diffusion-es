import numpy as np

file1 = open('cem_scores.txt', 'r')
Lines = file1.readlines()
scores = []
for line in Lines:
    scores.append(float(line.split()[1]))
scores = np.array(scores)
print('CEM')
print('mean', scores.mean())
print('std', scores.std())

file1 = open('gradient_scores.txt', 'r')
Lines = file1.readlines()
scores = []
for line in Lines:
    scores.append(float(line.split()[1]))
scores = np.array(scores)
print('Gradient')
print('mean', scores.mean())
print('std', scores.std())