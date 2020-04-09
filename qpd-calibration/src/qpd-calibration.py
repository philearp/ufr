# %%
def define_logger():
    '''
    The defined levels, in order of increasing severity, are the following:

        DEBUG <-- least severe
        INFO
        WARNING
        ERROR
        CRITICAL <-- most severe
    '''

    # Configure lowest logging level
    logging.basicConfig(level = logging.INFO)

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create handlers
    #c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('file.log')
    #c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    #c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    #logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info('Logging Started')

    return logger

def load_calibration_data(filepath):
    """Loads UFR tilt calibration data from file.

    Loads UFR tilt calibration data from .csv file and 
    returns the data as a pandas dataframe.

    Arguments (Required)
    --------------------
    filename : str
        Path to calibration file in form '/..../calibration_file.csv'

    Returns
    -------
    pandas dataframe
        5-column dataframe containing:
            theta_x - Optics tilt angle around x-axis (degrees).
            theta_y - Optics tilt angle around y-axis (degrees).
            qpd_x   - Signal from QPD: beam position in x-direction.
                      Normalised by qpd_sum so in range [-1 1].
            qpd_y   - Signal from QPD: beam position in y-direction.
                      Normalised by qpd_sum so in range [-1 1].
            qpd_sum - Signal from QPD: beam intensity.
    4-element tuple of float:
        Angular extent of calibration data.
        (theta_x.min, theta_x.max, theta_y.min, theta_y.max)

    Raises
    ------
    FileNotFoundError
        Raised if filename points 
    """

    col_names = ['theta_x', 'theta_y', 'qpd_x', 'qpd_y', 'qpd_sum']
    
    # Load the data
    try:
        df = pd.read_csv(filepath, names=col_names)
    except Exception as e:
        logger.error(f'Calibration could not be loaded', exc_info=True) # log exception
        raise # stop further execution

    # Verify data integrity
    try:
        assert df.shape[1] == 5, "calibration file must have 5 columns"
        assert all(i == float for i in df.dtypes), 'Calibration data must have type <float>'
    except Exception as e:
        logger.error("Assertion error occurred during data loading: ", exc_info=True)
        raise

    logger.info('Calibration datafile loaded successfully')

    angular_range = calc_xy_range(df['theta_x'], df['theta_y'])
    logger.info(f'Calibration data spans ({angular_range[0]} <= theta_x <= {angular_range[1]}) and ({angular_range[2]} <= theta_y <= {angular_range[3]}) (degrees)')

    return df, angular_range

def calc_xy_range(x, y):
    rng = (x.min(),
           x.max(),
           y.min(),
           y.max())
    return rng

def surface_fitting(x, y, z):
    """Fits 3rd order polynomial surface to z(x, y).

    Fits 3rd order polynomial function of form zfit = f(x) + g(y) to 
    scattered or gridded data of form z(x, y).

    Arguments (Required)
    --------------------
    x, y, z : array_like
              x, y, and z co-ordinates of the data points.
              May be 1D (e.g. scattered data) or 2D (e.g. gridded data).
              x, y and z must have same dimensions.

    Returns
    -------
    numpy.ndarray
        1D 7-element array containing the coefficients of:
            X^3, Y^3, X^2, Y^2, X, Y, const

    Raises
    ------
    LinAlgError
        Raised if computation of surface fit does not converge.
        Indicative of poor calibration data.
    """

    assert x.shape == y.shape, 'x and y arrays must have same dimensions'
    assert z.shape == x.shape, 'z array must have same dimension as x and y arrays'

    X = x.flatten()
    Y = y.flatten()
    Z = z.flatten()

    A = np.array([X**3, Y**3, X**2, Y**2, X, Y, X*0+1]).T

    coeff, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)
    
    return coeff

def fit_theta2qpd_surface(calibration_data):
    c = np.zeros([2, 7])

    logger.info('Fitting polynomial surface to qpd_x(theta_x, theta_y) data')
    c[0, :] = surface_fitting(calibration_data['theta_x'].to_numpy(), 
                            calibration_data['theta_y'].to_numpy(),
                            calibration_data['qpd_x'].to_numpy())
    logger.info(f'Fitting parameters: {c[0, :]}')

    logger.info('Fitting polynomial surface to qpd_y(theta_x, theta_y) data')
    c[1, :] = surface_fitting(calibration_data['theta_x'].to_numpy(),
                            calibration_data['theta_y'].to_numpy(),
                            calibration_data['qpd_y'].to_numpy())
    logger.info(f'Fitting parameters: {c[1, :]}')

    return c

def poly2Dreco(X, Y, c):
    """Evaluates 3rd order polynomial from coefficients

    Evaluates 3rd order polynomial function of form zfit = f(x) + g(y) to 
    scattered or gridded data of form z(x, y).

    Arguments (Required)
    --------------------
    X, Y : array_like
        Locations to evaluate the function: x and y co-ordinates.
        Can be 1D (e.g. scattered data) or 2d (e.g. gridded data)
    c : numpy.ndarray
        1D 7-element array containing the coefficients of:
            X^3, Y^3, X^2, Y^2, X, Y, const

    Returns
    -------
    numpy.ndarray
        Values of function at co-ordinates (x, y).
        Dimensions same as X, Y.
    """

    assert X.shape == Y.shape, 'x and y arrays must have same dimensions'
    assert c.shape == (7,), 'Incorrect dimensions for coefficent array'

    return (c[0]*X**3 + c[1]*Y**3 + c[2]*X**2 + c[3]*Y**2 + c[4]*X + c[5]*Y + c[6])

def plot_fitted_surface(x_raw, y_raw, z_raw, c, plot_opts):
    
    xy_lims = calc_xy_range(x_raw, y_raw)

    grid_x, grid_y = define_grid(xy_lims, 100)

    zfit = poly2Dreco(grid_x, grid_y, c)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=x_raw, y=y_raw, z=z_raw, mode='markers'))


    fig.add_trace(go.Surface(z=zfit, x=grid_x, y=grid_y))
                    
    fig.update_layout(scene = plot_opts)

    fig.update_layout(
    autosize=False,
    width=800,
    height=800)

    tx = plot_opts['xaxis_title']
    ty = plot_opts['yaxis_title']
    tz = plot_opts['zaxis_title']
    logger.info(f'Plotted figure: {tz}({tx}, {ty}) fitted surface.')

    # write figure
    fig_filename = f'{tz}({tx},{ty})-fitted-surface'
    fig.write_html(f'images/{fig_filename}.html')
    logger.info(f'Figure saved to "images/{fig_filename}.html"')

    if display_figures:
        fig.show()

def define_grid(xy_lims, nxy):
    """Creates gridded co-ordinates that span range of scattered co-ordinates.

    Generates a ny [#rows] x nx [#cols] grid of (x, y) co-ordinates within x and y
    limits defined by xy_lims.

    x ranges from xy_lims(0) to xy_lims(1)
    y ranges from xy_lims(1) to xy_lims(2)

    nxy can be a single integer e.g. 100 in which case nx = ny = nxy
                              or
    nxy can be a tuple of two ints in format (nx, ny)

    Arguments (Required)
    --------------------
    xy_lims : 4-element tuple of float
        (x_min, x_max, y_min, y_max)

    nxy : int or tuple of 2 ints
        int: equal number of points in x and y directions, nx = ny.
        tuple of 2 ints: nx = nxy[0]
                         ny = nxy[1]

    Returns
    -------
    numpy.ndarray
        Arrays containing grid of x co-ordinates and y co-ordinates
    """
    assert type(nxy) == int or type(nxy) == tuple, 'nxy must be int or (int, int)'
    if type(nxy) == tuple:
        assert len(nxy) == 2, 'nxy as tuple must contain 2 integer values only'
        assert all(type(i) == int for i in nxy), 'all elements of nxy must be of type int'
        nx = nxy[0]
        ny = nxy[1]
        assert nx > 0, 'nx must be positive'
        assert ny > 0, 'ny must be positive'
    else: # type(nxy) == int
        assert nxy > 0, 'nxy must be positive'
        nx = nxy
        ny = nxy

    # Calculate upper and lower extent of grid   
    x_min = xy_lims[0]
    x_max = xy_lims[1]
    y_min = xy_lims[2]
    y_max = xy_lims[3]

    # Define step size (complex number means 'this many steps' in np.mgrid)
    x_step = nx * 1j
    y_step = ny * 1j

    # Generate grid
    grid_x, grid_y = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step]

    return grid_x, grid_y

def plot_fitted_theta2qpd_surfaces(calibration_data, c):
    # Plot variation of QPD x-position with theta_x and theta_y
    plot_opts = dict(
        xaxis_title='theta_x',
        yaxis_title='theta_y',
        zaxis_title='QPD_x')
    plot_fitted_surface(calibration_data['theta_x'],
                        calibration_data['theta_y'],
                        calibration_data['qpd_x'],
                        c[0, :],
                        plot_opts)
    

    # Plot variation of QPD y-position with theta_x and theta_y
    plot_opts = dict(
        xaxis_title='theta_x',
        yaxis_title='theta_y',
        zaxis_title='QPD_y')
    plot_fitted_surface(calibration_data['theta_x'],
                        calibration_data['theta_y'],
                        calibration_data['qpd_y'],
                        c[1, :],
                        plot_opts)
    

def calc_angles_from_qpd_values(c, qpd_pos, calc_opts):
    """Calibration function - converts QPD position to angles 

    Converts QPD position (qpd_x, qpd_y) to angles (theta_x, theta_y)

    Calculates angles theta_x and theta_y that satisfy:
        [A] fitted surface qpd_x(theta_x, theta_y) == qpd_pos[0]
        [B] fitted surface qpd_y(theta_x, theta_y) == qpd_pos[1]
    Each takes the form of a line in the (theta_x, theta_y) plane
    
        Each of [A] and [B] are calculated numerically as the intersection between:
            [i] the polynomial surface z = poly2Dreco(theta_x, theta_y, c)
                where 'c' are the parameters identified in surface_fitting()
            [ii] the plane z = <qpd value>
        The calculation is evaluated in the same angular extent as the calibration
        data to ensure that we are not extrapolating the fitting functions outside the
        calibration data region.

    The 'common intersection' between the two calculated lines in the
    (theta_x, theta_y) plane is calculated. This identifies the pair of 
    (theta_x, theta_y) values which satisfy equations [A] and [B].

    Arguments (Required)
    --------------------
   c : (2, 7) numpy.ndarray
        2 x 7 array containing the coefficients of:
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_x(theta_x, theta_y)
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_y(theta_x, theta_y)
    qpd_pos : 2-element tuple of float
        Containing QPD position for evaluation (qpd_x, qpd_y).
    calc_opts : dict containing
        angular_range : 4-element tuple of float
            Angular extent of calibration data
            (theta_x.min, theta_x.max, theta_y.min, theta_y.max)
        intersection_threshold : float
        calculation_grid_step_size : float
    
    Returns
    -------
    2-element tuple of float
        <arg1 description>
    """

    assert type(c) == np.ndarray, 'c must be of type numpy.ndarray'
    assert c.shape == (2, 7), 'c must be of shape (2, 7)'

    assert type(qpd_pos) == tuple, 'qpd_pos must be of type tuple'
    assert len(qpd_pos) == 2, 'qpd_pos must contain 2 elements'
    assert all(type(i) == np.float64 or type(i) == float for i in qpd_pos), 'qpd_pos must contain elements of type int or float'

    # unpack dict
    calculation_grid_step_size = calc_opts['calculation_grid_step_size']
    angular_range = calc_opts['angular_range']

    assert type(angular_range) == tuple, 'angular_range must be of type tuple'
    assert len(angular_range) == 4, 'angular_range must contain 2 elements'
    assert all(type(i) == int or type(i) == float for i in angular_range), 'angular_range must contain elements of type int or float'

    assert type(calculation_grid_step_size) == float, 'calculation_grid_step_size must be of type float'
    assert calculation_grid_step_size > 0, 'calculation_grid_step_size must be positive'

    # [1] Define grid of (theta_x, theta_y). 
    grid_tx, grid_ty = np.meshgrid(np.arange(angular_range[0],
                                             angular_range[1],
                                             calculation_grid_step_size),
                                   np.arange(angular_range[2],
                                             angular_range[3],
                                             calculation_grid_step_size))

    # [2] Identify as True the (theta_x, theta_y) values which generate QPD position 'qpd_pos'
    is_common_intersection = calc_common_intersection(grid_tx, grid_ty, c, qpd_pos, calc_opts)
    
    # [3] Extract mean theta_x and theta_y from True values in is_common_intersection array
    thetas = calc_angles_from_common_intersection(grid_tx, grid_ty, is_common_intersection)

    # [4] Validate calculation validity
    calculation_validity = verify_qpd_from_angles(thetas, c, qpd_pos, calc_opts)
    
    try:
        assert calculation_validity, 'Calculation of thetas from QPD values produced invalid result'
    except Exception as e:
        logger.warning(f'Invalid calculated angles for QPD position ({qpd_pos[0]}, {qpd_pos[1]})', exc_info=False)
        #raise

    return (thetas, calculation_validity)

def calc_common_intersection(grid_tx, grid_ty, c, qpd_pos, calc_opts):
    """Numerical calculation of plane-surface intersections

    Numerically calculates the common intersection between:
        [a] the intersection between:
            [i] the polynomial surface qpd_x: z = poly2Dreco(grid_tx, grid_ty, c[0, :])
                where 'c' are the parameters identified in surface_fitting()
            [ii] the plane z = qpd_x_position
        [b] the intersection between:
            [i] the polynomial surface qpd_y: z = poly2Dreco(grid_tx, grid_ty, c[1, :])
                where 'c' are the parameters identified in surface_fitting()
            [ii] the plane z = qpd_y_position
    
    The calculation is evaluated in the same angular extent as the calibration
    data to ensure that we are not extrapolating the fitting functions outside the
    calibration data region.

    Arguments (Required)
    --------------------
    grid_tx, grid_ty : (r, c) numpy.ndarray 
        Grid of co-ordinates for evaluation of polynomial surface. Spans the angular 
        range of (theta_x, theta_y) used in the calibration.
    c : (2, 7) numpy.ndarray
        2 x 7 array containing the coefficients of:
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_x(theta_x, theta_y)
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_y(theta_x, theta_y)
    qpd_pos : 2-element tuple of float
        Containing QPD position for evaluation (qpd_x_position, qpd_y_position).
    calc_opts : dict containing
        intersection_threshold : float

    Returns
    -------
    (r, c) numpy.ndarray array of bool
        Logical array:
            True - where (theta_x, theta_y) satisfy the 'common intersection'
                   described above.
            False - in all other positons
    """

    intersection_array = np.zeros((2,) + grid_tx.shape) # initialise

    intersection_threshold = calc_opts['intersection_threshold']

    for i in range(0, 2): # For qpd_x and qpd_y,
        # Calculate plane-surface intersection
        z_qpd = poly2Dreco(grid_tx, grid_ty, c[i, ]) - qpd_pos[i] # z_qpd_x == 0 where qpd_x == qpd_pos[0]
        is_intersection = np.abs(z_qpd - 0) < intersection_threshold

        intersection_array[i,] = is_intersection

    # Calculate common intersection (i.e. find theta_x and theta_y position)
    is_common_intersection = np.all(intersection_array, axis=0)
    
    return is_common_intersection

def calc_angles_from_common_intersection(grid_tx, grid_ty, is_common_intersection):
    """Calculates angle from logical array

    Takes logical array 'is_common_intersection' and looks up mean value
    of theta_x and theta_y from the arrays grid_tx and grid_ty where 
    is_common_intersection is True.

    Arguments (Required)
    --------------------
    grid_tx, grid_ty : (r, c) numpy.ndarray of float
        Angular co-ordinates (theta_x, theta_y)
    is_common_intersection : (r, c) numpy.ndarray of bool
        Contains True values corresponding to the numerically calculated
        angles (theta_x, theta_y)

    Returns
    -------
    2-element tuple of float
        Calculated angles (theta_x, theta_y)
    """

    theta_x = np.mean(grid_tx[is_common_intersection])
    theta_y = np.mean(grid_ty[is_common_intersection])

    thetas = (theta_x, theta_y)

    return thetas

def verify_qpd_from_angles(thetas, c, qpd_pos, calc_opts):
    """Verifies calculated angles by back-calculating QPD position

    Back-calculates QPD position from the numerically calculated angles
    and compares to measured QPD positions.

    Calculates 'back-calculation error' as:
    error = back_calculated_qpd_position - measured_qpd_position

    Calculation is 'valid' if abs(error) < verification_threshold

    Arguments (Required)
    --------------------
    thetas : 2-element tuple of float
        Calculated angles (theta_x, theta_y)
    c : (2, 7) numpy.ndarray
        2 x 7 array containing the coefficients of:
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_x(theta_x, theta_y)
            X^3, Y^3, X^2, Y^2, X, Y, const for qpd_y(theta_x, theta_y)
    qpd_pos : 2-element tuple of float
        Containing QPD position for evaluation (qpd_x, qpd_y).
    calc_opts : dict containing
        verification_threshold : float

    Returns
    -------
    bool
        True if error < verification_threshold
        False if error > verification_threshold
    """

    verification_threshold = calc_opts['verification_threshold']
    assert type(verification_threshold) == float, 'verification_threshold must be type float'
    assert verification_threshold > 0, 'verification_threshold must be positive'

    qpd_chk = np.zeros_like(qpd_pos)
    qpd_chk[0] = poly2Dreco(thetas[0], thetas[1], c[0, ])
    qpd_chk[1] = poly2Dreco(thetas[0], thetas[1], c[1, ])

    if all((qpd_chk[i] - qpd_pos[i]) < verification_threshold for i in range(2)):
        calculation_validity = True
    else:
        calculation_validity = False

    # print('input qpd position: ' + str(qpd_pos))
    # print('calculated angles are ' + str(thetas) + ' degrees')   
    # print('verification qpd position: ' + str([qpd_chk[0], qpd_chk[1]]))

    return calculation_validity


''' %%
qpd_pos = (0., 0.09) # tuple of float

calculation_options = dict(
            angular_range=calibration_data_angular_range,
            calculation_grid_step_size=0.01,
            intersection_threshold=0.01,
            verification_threshold=0.01)

thetas_numerical, calculation_validity = calc_angles_from_qpd_values(c, qpd_pos, calculation_options)
print(calculation_validity) '''

def iterate_qpd_positions(grid_qx, grid_qy, c, calc_opts):
    logger.info(f'Starting iteration through grid of QPD positions...')
    logger.info('   For each QPD position, calculate (theta_x, theta_y).')

    tx_array = np.zeros_like(grid_qx)
    ty_array = np.zeros_like(grid_qx)

    validity = np.zeros_like(grid_qx)
    
    for i, j in np.ndindex(grid_qx.shape):
        qpd_pos = (grid_qx[i, j], grid_qy[i, j])

        t, v = calc_angles_from_qpd_values(c, qpd_pos, calc_opts)
        
        tx_array[i, j] = t[0]
        ty_array[i, j] = t[1]

        validity[i, j] = v
        
    logger.info(f'Completed iteration through grid of QPD positions.')

    if validity.all():
        logger.info('Calculations for all QPD positions produced valid results')
    else:
        logger.error('One or more QPD positions produced invalid calculated angles')
    
    return tx_array, ty_array, validity.all()

def fit_qpd2theta_surface(grid_qx, grid_qy, tx_array, ty_array):
    d = np.zeros([2, 7])

    logger.info('Fitting polynomial surface to theta_x(qpd_x, qpd_y) data')
    d[0, :] = surface_fitting(grid_qx.flatten(), grid_qy.flatten(), tx_array.flatten())
    logger.info(f'Fitting parameters: {d[0, :]}')

    logger.info('Fitting polynomial surface to theta_y(qpd_x, qpd_y) data')
    d[1, :] = surface_fitting(grid_qx.flatten(), grid_qy.flatten(), ty_array.flatten())
    logger.info(f'Fitting parameters: {d[1, :]}')

    return d

def main_calibration():
    
    logger.info(f'Start calibration using file: "{filename}"')
    
    calibration_data, calibration_data_angular_range = load_calibration_data(filename)
    #print(calibration_data)

    # Fit surface to qpd(theta) data
    c = fit_theta2qpd_surface(calibration_data)

    # Plot fitted theta2qpd surface (optional)
    if plot_figures:
        plot_fitted_theta2qpd_surfaces(calibration_data, c)

    # iterate through QPD positions
    qx_vec = np.arange(-0.3, 0.3, 0.01)
    qy_vec = np.arange(-0.5, 0.1, 0.01)
    grid_qx, grid_qy = np.meshgrid(qx_vec, qy_vec)

    calculation_options = dict(
                angular_range=calibration_data_angular_range,
                calculation_grid_step_size=0.01,
                intersection_threshold=0.01,
                verification_threshold=0.01)

    tx_array, ty_array, calculation_validity = iterate_qpd_positions(grid_qx, grid_qy, c, calculation_options)

    # Fit surface to theta(qpd) data
    d = fit_qpd2theta_surface(grid_qx, grid_qy, tx_array, ty_array)

    # Plot fitted qpd2theta surface (optional)
    if plot_figures:
        plot_fitted_qpd2theta_surfaces(grid_qx, grid_qy, tx_array, ty_array, d)

    logger.info('Calibration completed successfully.')

    return d

def plot_fitted_qpd2theta_surfaces(grid_qx, grid_qy, tx_array, ty_array, d):

    # Plot variation of $\theta_x$ with $q_x$ and $q_y$
    plot_opts = dict(
                xaxis_title='QPD_x',
                yaxis_title='QPD_y',
                zaxis_title='theta_x')
    plot_fitted_surface(grid_qx.flatten(), grid_qy.flatten(), tx_array.flatten(), d[0, :], plot_opts)

    # Plot variation of $\theta_y$ with $q_x$ and $q_y$
    plot_opts = dict(
                xaxis_title='QPD_x',
                yaxis_title='QPD_y',
                zaxis_title='theta_y')
    plot_fitted_surface(grid_qx.flatten(), grid_qy.flatten(), ty_array.flatten(), d[1, :], plot_opts)

def qpd2angle(qpd_pos, d):  
    
    # convert tuple --> np.ndarray for use in poly2Dreco
    qpd_x = np.array([qpd_pos[0]])
    qpd_y = np.array([qpd_pos[1]])

    theta_x = poly2Dreco(qpd_x, qpd_y, d[0, :])
    theta_y = poly2Dreco(qpd_x, qpd_y, d[1, :])

    return (theta_x[0], theta_y[0])

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.express as px
import logging
import sys

plot_figures = True
display_figures = False
filename = '../data/2020-03-10_QPD-Tilt-Calibration_Test1.csv'

logger = define_logger()

d = main_calibration()

# %%
#qpd_pos = (0.01, 0.1)
#print(qpd2angle(qpd_pos, d))

