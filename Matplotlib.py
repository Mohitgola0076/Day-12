                # Using style sheets :
'''                
The style package adds support for easy-to-switch plotting "styles" with the same parameters as a matplotlib rc file (which is read
at startup to configure Matplotlib).
There are a number of pre-defined styles provided by Matplotlib. For example, there's a pre-defined style called "ggplot", which 
emulates the aesthetics of ggplot (a popular plotting package for R). To use this style, just add:
'''
            # Example : 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
plt.style.use('ggplot')
data = np.random.randn(50)
To list all available styles, use:

print(plt.style.available)

        # Output :

['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblin']

#############################################################################################################################

                    # Matplotlib.pyplot.subplots() in Python : 

Matplotlib is a library in Python and it is numerical – mathematical extension for NumPy library. Pyplot is a state-based interface to a Matplotlib module which provides a MATLAB-like interface.

            # sample code
import matplotlib.pyplot as plt
	
plt.plot([1, 2, 3, 4], [16, 4, 1, 8])
plt.show()


                    # matplotlib.pyplot.subplots() Function :
                    
The subplots() function in pyplot module of matplotlib library is used to create a figure and a set of subplots.
Syntax: matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, 
**fig_kw)

                    # Example : 

# Implementation of matplotlib function
import numpy as np
import matplotlib.pyplot as plt

# First create some toy data:
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x**2)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')


fig.suptitle('matplotlib.pyplot.subplots() Example')
plt.show()

                    # Example : 

# Implementation of matplotlib function
import numpy as np
import matplotlib.pyplot as plt

# First create some toy data:
x = np.linspace(0, 1.5 * np.pi, 100)
y = np.sin(x**2)+np.cos(x**2)

fig, axs = plt.subplots(2, 2,
						subplot_kw = dict(polar = True))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

fig.suptitle('matplotlib.pyplot.subplots() Example')
plt.show()

###############################################################################################################################

                # Multiple Plots using subplot () Function : 
A subplot () function is a wrapper function which allows the programmer to plot more than one graph in a single figure by just calling it once.
Syntax: matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)

                        # Example : 

# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the angles from 0 to 2 pie (360 degree) in narray object
X = np.arange(0, math.pi*2, 0.05)

# Using built-in trigonometric function we can directly plot
# the given cosine wave for the given angles
Y1 = np.sin(X)
Y2 = np.cos(X)
Y3 = np.tan(X)
Y4 = np.tanh(X)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)

# For Sine Function
axis[0, 0].plot(X, Y1)
axis[0, 0].set_title("Sine Function")

# For Cosine Function
axis[0, 1].plot(X, Y2)
axis[0, 1].set_title("Cosine Function")

# For Tangent Function
axis[1, 0].plot(X, Y3)
axis[1, 0].set_title("Tangent Function")

# For Tanh Function
axis[1, 1].plot(X, Y4)
axis[1, 1].set_title("Tanh Function")

# Combine all the operations and display
plt.show()

############################################################################################################################


                    # matplotlib.pyplot.figure() Function :
The figure() function in pyplot module of matplotlib library is used to create a new figure.

        # Syntax: 
matplotlib.pyplot.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True, FigureClass=, 
clear=False, **kwargs)

        # Parameters: 
This method accept the following parameters that are described below:

num : This parameter is provided, and a figure with this id already exists.
figsize(float, float): These parameter are the width, height in inches.
dpi : This parameter is the resolution of the figure.
facecolor : This parameter is the the background color.
edgecolor : This parameter is the border color.
frameon : This parameter suppress drawing the figure frame.
FigureClass : This parameter use a custom Figure instance.
clear : This parameter if True and the figure already exists, then it is cleared.

                # Example : 

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import matplotlib.lines as lines


fig = plt.figure()
fig.add_artist(lines.Line2D([0, 1, 0.5], [0, 1, 0.3]))
fig.add_artist(lines.Line2D([0, 1, 0.5], [1, 0, 0.2]))

plt.title('matplotlib.pyplot.figure() Example\n',
				fontsize = 14, fontweight ='bold')

plt.show()
