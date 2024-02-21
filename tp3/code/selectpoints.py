#!/usr/bin/env python
# -*- noplot -*-

"""
This example shows how to use matplotlib to provide a data cursor.
It uses matplotlib to draw the cursor and may be slow since this requires
redrawing the figure with every mouse move.

Faster cursoring is possible using native GUI drawing, as in wxcursor_demo.py
"""

from __future__ import print_function
from pylab import *
from skimage import io

class Cursor:
    def __init__(self, ax, s):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.f = "C:/Users/Haba/Downloads/imag/haba_mathieu.txt"  # Output file path
        self.count = 1
        self.s = s

    def mouseclick(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        print(x, y)
        # Append the coordinates to the output file
        with open(self.f, 'a') as h:
            h.write("\t{}\t{}\n".format(x, y))

        # Display the point number at the clicked location
        self.ax.text(x + 4, y - 4, str(self.count), fontsize=14, color='r')
        self.ax.plot(x, y, '.r')
        self.count += 1
        draw()

# Load the image
image_path = "C:/Users/Haba/Downloads/imag/haba_mathieu.jpg"
image = io.imread(image_path)

# Create a Matplotlib figure and axes to display the image
fig, ax = subplots()
ax.imshow(image)

# Create an instance of the Cursor class
cursor = Cursor(ax, image.shape)
# Connect the button_press_event event to the mouseclick method
connect('button_press_event', cursor.mouseclick)

# Display the image
show()
