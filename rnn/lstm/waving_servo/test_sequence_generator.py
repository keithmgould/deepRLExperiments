from sequence_generator import SequenceGenerator

# how long should the entire sequence take (in ms)
SEQUENCE_DURATION_MS = 4000

# Noon, 3, Noon, 3, Noon, 9, Noon, 9
# SEQUENCE = [0, 90, 0, 90, 0, -90, 0, -90]
SEQUENCE = [0,90]

sg = SequenceGenerator(SEQUENCE_DURATION_MS, SEQUENCE)

print("time,\tstart,\tend, \t%, \tpos")

for x in range(8000):
  sg.position(x)