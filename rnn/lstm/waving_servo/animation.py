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
oval = c.create_oval(5,5,60,60,fill='black')
start_point_x = 400
start_point_y = 700
line_length = 300
line = c.create_line(
  start_point_x,
  start_point_y,
  start_point_x,
  start_point_y - line_length,
  fill="black")

xd = 5
yd = 10

gui.title("Neato")

sg = SequenceGenerator(SEQUENCE_DURATION_MS, SEQUENCE)

start_time = time.time()

def convert_rad_to_x_y(degrees, length):
  x = math.sin(math.radians(degrees)) * length
  y = math.cos(math.radians(degrees)) * length
  return x,y

while True:
  c.move(oval,xd,yd)
  elapsed_time = (time.time() - start_time) * 1000
  angle = sg.position(elapsed_time)
  x,y = convert_rad_to_x_y(angle, line_length)
  # print("{},\t{},\t{},\t{}".format(elapsed_time, angle,x,y))
  c.coords(
    line,
    start_point_x,
    start_point_y,
    start_point_x + x,
    start_point_y - y)

  p=c.coords(oval)
  if p[3] >= 800 or p[1] <=0:
     yd = -yd
  if p[2] >=800 or p[0] <=0:
     xd = -xd

  
  gui.update()
  time.sleep(0.025)import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegFileWriter

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r', animated=True)
f = np.linspace(-3, 3, 200)


def init():
  ax.set_xlim(-3, 3)
  ax.set_ylim(-0.25, 2)
  ln.set_data(xdata,ydata)
  return ln,

def update(frame):
  xdata.append(frame)
  ydata.append(np.exp(-frame**2))
  ln.set_data(xdata, ydata)
  return ln,


ani = FuncAnimation(fig, update, frames=f,
                    init_func=init, blit=True, interval = 2.5,repeat=False)
plt.show()