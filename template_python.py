import numpy as np
import matplotlib.pyplot as plt

from phas0007anim_v2 import animate_trajectory_2d, animate_trajectories_2d
from assignment_tests import test_force, test_move_planet, test_trajectory_return_type

G       = 6.6743e-11 # gravitational constant [m^3 kg^-1 s^-2]
YEAR    = 3.154e7    # one year in seconds
M_SOLAR = 1.989e30   # solar mass [kg]
AU      = 1.496e11   # one astronomical unit (mean orbital radius of Earth) [m]

def force(pos1, pos2, m1, m2):
    """
    Calculate the gravitational force exerted by object 1 on object 2.
    
    Input:
        pos1: position vector of first object  (NumPy array)
        pos2: position vector of second object (NumPy array)
        m1:   mass of first object
        m2:   mass of second object

    Depends on:
        G  (global variable):    gravitational constant [m^3 kg^-1 s^-2]

    Return:
        force on object 2 [N] (NumPy array)
    """
    ##################
    # YOUR CODE HERE #
    ##################
    global G

    r=pos2-pos1
    r_mag=np.linalg.norm(r)

    force=-G*m1*m2*r/r_mag**3

    return force
    
def move_planet(position, velocity, m_star, dt):
    """
    Calculate motion of planet in the gravitational field of a star with given mass
    at the originin two dimensions, using Euler's method.
    
    Input:
      - position: position vector of planet at start of time step (NumPy array: x, y)
      - velocity: velocity vector of planet at start of time step (NumPy array: vx, vy)
      - m_star:   mass of star
      - dt:       time step
      
    Output: (position_new, velocity_new)
      - position_new: position vector of planet at end of time step (NumPy array: x, y)
      - velocity_new: velocity vector of planet at end of time step (NumPy array: vx, vy)
      
    Depends on:
      - force = function to calculate the gravitational force between two objects
    """
    ##################
    # YOUR CODE HERE #
    ##################

    star_pos=np.array([0.0,0.0])
    acc=force(star_pos,position,m_star,1)
    pos_new=position+velocity*dt
    vel_new=velocity+acc*dt

    return pos_new,vel_new


def trajectory(position, velocity, m_star, dt, t_max):
    """
    Calculate trajectory of planet from given starting position in two dimensions, using Euler's method.
    
    Input:
      - position: position vector of planet at start of simulation [m] (NumPy array: x, y)
      - velocity: velocity vector of planet at start of simulation [m] (NumPy array: vx, vy)
      - m_star:   mass of star [kg]
      - dt:       time step    [s]
      - t_max:    duration of calculated motion [s]

    Returns:
        Trajectory as a 2 x N array containing x and y coordinates at each instant.
    """
    ##################
    # YOUR CODE HERE #
    ##################

    t=0
    x,y=position
    x_arr,y_arr=[x],[y]
    
    while t<=t_max:
        t+=dt
        position,velocity=move_planet(position, velocity, m_star, dt)
        x,y=position
        x_arr.append(x)
        y_arr.append(y)

    ans=np.array([x_arr,y_arr])

    return ans


def plot_trajectory(x, y, label="trajectory", fmt='-'):
    """Plot trajectory with equal aspect ratio, using existing figure.

    Arguments:
        x, y:  arrays of x- and y-coordinates
        label: (optional) label for use in legend
        fmt:   (optional) format to use for plotting, e.g. ':' for dotted line
    """
    plt.plot(x,y,fmt,label=label)
    plt.gca().set_aspect('equal') # same scale horizontally and vertically
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

