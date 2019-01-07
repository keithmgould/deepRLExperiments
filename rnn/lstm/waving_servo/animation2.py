#!/usr/bin/python
from tkinter import *
import time
import math
import numpy as np
from sequence_generator import SequenceGenerator

SEQUENCE_DURATION_MS = 4000

# Noon, 3, Noon, 3, Noon, 9, Noon, 9
SEQUENCE = [0, 90, 0, 90, 0, -90, 0, -90]
# SEQUENCE = [0,90]

gui = Tk()
gui.geometry("800x800")
c = Canvas(gui ,width=800 ,height=800)
c.pack()

start_point_x = 400
start_point_y = 700
line_length = 300
line = c.create_line(
  start_point_x,
  start_point_y,
  start_point_x,
  start_point_y - line_length,
  fill="black")

gui.title("Servo Waver")

sg = SequenceGenerator(SEQUENCE_DURATION_MS, SEQUENCE)

start_time = time.time()

def convert_rad_to_x_y(degrees, length):
  x = math.sin(math.radians(degrees)) * length
  y = math.cos(math.radians(degrees)) * length
  return x,y

while True:
  elapsed_time = (time.time() - start_time) * 1000
  angle = sg.position(elapsed_time)
  x,y = convert_rad_to_x_y(angle, line_length)
  c.coords(
    line,
    start_point_x,
    start_point_y,
    start_point_x + x,
    start_point_y - y)

  gui.update()
  time.sleep(0.025)