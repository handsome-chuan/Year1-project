import numpy as np
from numpy.linalg import norm

G = 6.6743e-11 # gravitational constant [m^3 kg^-1 s^-2]

def test_force(force):
    """
    Check whether force function gives correct results in some simple cases,
    and print report.

    Parameters
    ----------
    force: the function to be tested
    """
    test_fn_force(force,"distance = 1 in x direction",(0,0),(1,0),1,1,(-G,0))
    test_fn_force(force,"swap objects",(1,0),(0,0),1,1,(G,0))
    test_fn_force(force,"distance = 2",(0,0),(2,0),1,1,(-0.25*G,0))
    test_fn_force(force,"distance = 1 in y direction",(0,0),(0,1),1,1,(0,-G))
    test_fn_force(force,"displaced from origin",(10,0),(10,1),1,1,(0,-G))
    test_fn_force(force,"non-unit mass 1",(0,0),(1,0),2,1,(-2*G,0))
    test_fn_force(force,"non-unit mass 2",(0,0),(1,0),1,2,(-2*G,0))

def test_move_planet(move_planet):
    """
    Check whether move_planet function gives correct results in some simple cases,
    and print report.

    Parameters
    ----------
    move_planet: the function to be tested
    """
    # Some arbitrary but useful quantities for test scenarios
    R = 1e10   # distance
    T = 1e5    # time
    V = 2*R/T  # speed
    M = 1e28   # mass
    test_fn_move_planet(move_planet, "zero time-step", (R,0), (V,0), M, 0, (R,0), (V,0))
    test_fn_move_planet(move_planet, "planet starts at rest", (R,0), (0,0), M, T, (R,0), (-G*M*T/R**2,0))
    test_fn_move_planet(move_planet, "planet moving away from star", (R,0), (V,0), M, T, (R+V*T,0), (V-G*M*T/R**2,0))  
    test_fn_move_planet(move_planet, "planet moving towards star", (R,0), (-V,0), M, T, (R-V*T,0), (-V-G*M*T/R**2,0))  
    test_fn_move_planet(move_planet, "planet moving past star", (R,0), (0,V), M, T, (R, V*T), (-G*M*T/R**2, V))
    test_fn_move_planet(move_planet, "smaller time-step", (R,0), (0,V), M, T/10, (R,V*T/10), (-G*M*T/10/R**2,V))
    test_fn_move_planet(move_planet, "larger stellar mass", (R,0), (0,V), M*5, T, (R,V*T), (-G*M*5*T/R**2,V))

def test_trajectory_return_type(trajectory):
    """
    Check whether trajectory function correctly returns a 2 x N NumPy array.
    """
    pos = np.array([1,1])
    vel = np.array([1,1])
    traj = trajectory(pos,vel,1,1,1)
    if not isinstance(traj, np.ndarray):
        print(f"ERROF: function should return a NumPy array but returns type: {type(traj)}")
        return
    dimensions = np.shape(traj)
    n_dim = len(dimensions)
    if n_dim != 2:
        print(f"ERROR: returned array should have 2 dimensions but has {n_dim} dimensions with shape {dimensions}")
        return
    if dimensions[0] != 2:
        print(f"ERROR: first dimension of array should be 2 but is {dimensions[0]}: shape is {dimensions}")
        return
    print("OK: function return array with dimensions 2 x N")

def test_fn_move_planet(move_planet, description, position, velocity, m_star, dt, expected_pos, expected_vel):
    """Check whether move_planet function gives expected results.
    
    Parameters
    ----------
    move_planet:                    the function to be tested
    description:                    summary of the scenario being tested
    position, velocity, m_star, dt: input to function under test
    expected_pos, expected_vel:     output expected if function under test is correct
    """
    epsilon = 1e-10
    results = move_planet(np.array(position), np.array(velocity), m_star, dt)
    if not isinstance(results, tuple):
        print(f"ERROR: function should return two vectors but returns {results}.")
        return
    if not len(results)==2:
        print(f"ERROR: function should return two vectors but returns {results}.")
        return
    pos, vel = results
    if not (isinstance(pos,np.ndarray) and isinstance(vel,np.ndarray)):
        print(f"ERROR: function should return two vectors but returns {results}.")
        return
    args_as_string = f"{position}, {velocity}, {m_star}, {dt}"
    err_pos, err_vel = norm(pos - expected_pos), norm(vel - expected_vel)
    if err_pos < epsilon and err_vel < epsilon:
        print(f"OK: correct results for test '{description}'")
    else:
        print(f"ERROR: wrong results for test '{description}'")
        print(f"  input:    {args_as_string}")
        print(f"  expected: {expected_pos, expected_vel}")
        print(f"  got:      {results}")

def test_fn_force(force, description, pos1, pos2, m1, m2, expected_force):
    """
    Check whether force function gives expected results and print report.

    Parameters
    ----------
    force:              the function to be tested
    description:        summary of the scenario being tested
    pos1, pos2, m1, m2: input to function under test
    expected_force:     output expected if function under test is correct
    """
    # We expect results of order G since distances and masses are of order one
    # so this is roughly equivalent to relative error of 1e-6.
    epsilon = 1e-6 * G
    f = force(np.array(pos1), np.array(pos2), m1, m2)
    if not isinstance(f,np.ndarray):
        print(f"ERROR: function should return a vector but returns {f}.")
        return
    args_as_string = f"({pos1}, {pos2}, {m1}, {m2})"
    error = norm(f-expected_force)
    if error<epsilon:
        print(f"OK: correct results for test '{description}'")
    else:
        print(f"ERROR: wrong results for test '{description}'")
        print(f"  input:    {args_as_string}")
        print(f"  expected: {expected_force}")
        print(f"  got:      {f}")