import math

# the servo waves with the following pattern:
#
# Noon, 3, Noon, 3, Noon, 9, Noon, 9 (then sequence repeats)
#
# To generate the sequence the below function takes time (in milliseconds) as an argument,
# and outputs angle (in radians), where the positive direction is clockwise, 
# and zero is at “Noon.”

# how long should the entire sequence take (in ms)
SEQUENCE_DURATION_MS = 4000

# Noon, 3, Noon, 3, Noon, 9, Noon, 9
SEQUENCE = [0, 1.5708, 0, 1.5708, 0, -1.5708, 0, -1.5708]

class SequenceGenerator:

  def __init__(self, sequence_duration, sequence):
    self.sequence_duration = sequence_duration
    self.sequence = sequence
    self.sequence_step_count = len(self.sequence)
    self.time_between_steps = self.sequence_duration / self.sequence_step_count

  def position(self, time):
    modulus_time = time % self.sequence_duration
    percent_complete_of_sequence = modulus_time / self.sequence_duration
    percent_complete_of_step = (modulus_time % self.time_between_steps) / self.time_between_steps
    start_step = math.floor(percent_complete_of_sequence * self.sequence_step_count)
    end_step = (start_step + 1) % self.sequence_step_count
    step_size = math.fabs(start_step - end_step)
    start_value = self.sequence[start_step]
    end_value = self.sequence[end_step]

    if start_value > end_value:
      pos = start_value - step_size * percent_complete_of_step
    else:
      pos = start_value + step_size * percent_complete_of_step

    # print("{},\t{},\t{},\t{},\t{}". format(time, start_value, end_value, percent_complete_of_step, pos))
    return pos

sg = SequenceGenerator(SEQUENCE_DURATION_MS, SEQUENCE)

# print("time,\tstart,\tend, \t%, \tpos")
# sg.position(100)
# sg.position(600)
# sg.position(1100)
# sg.position(1600)
# sg.position(2100)
# sg.position(2600)
# sg.position(3100)
# sg.position(3600)

# print("--------------------")
# sg.position(4100)
# sg.position(4600)
# sg.position(4100)
# sg.position(4600)
# sg.position(6100)
# sg.position(6600)
# sg.position(7100)
# sg.position(7600)