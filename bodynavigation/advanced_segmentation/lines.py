from loguru import logger
import math
import numpy as np
import bodynavigation


def standard_from_slopeintercept(angle, point):
    # TODO odstranit závislost na sympy
    from sympy import symbols, Eq, solve

    """Transfer a line's slope-intercept formulation to standard form (ax + by + c = 0).

    Args:
        angle (float): the angle between x axis and the line, in degrees
        point (list [x coordinate, y coordinate]): any point on the line

    Returns:
        a, b, c: the line's standard formcoeficients
    """
    an = 180 - angle
    point = np.asarray(point).tolist()

    a1 = math.cos(math.radians(an))
    a2 = math.sin(math.radians(an))
    x0 = point[0]
    y0 = point[1]

    t = a2 / a1
    a2 = t * a2
    x0 = t * x0
    # TODO Tohle dokážeme spočítat i na papíře bez bez balíku sympy.
    x, y = symbols("x y")
    eq = Eq((-1 * y) + (-1 * x * t) + (x0) + (y0), 0)
    result = solve(eq)
    print(result)

    a = 1
    # Třeba je to správně, ale vypadá to děsivě.
    if point == [0, 0]:
        c = 0
        if angle == 90 or angle == 270:
            b = 0
        elif angle == 0 or angle == 360 or angle == 180:
            a = 0
            b = 1
        else:
            b = -1 * result[0][x].args[0]
    elif angle == 90 or angle == 270:
        b = 0
        c = -1 * result[0][x].args[0]
    elif angle == 0 or angle == 360 or angle == 180:
        a = 0
        b = 1
        c = -1 * point[1]
    else:
        b = -1 * result[0][x].args[1].args[0]
        c = -1 * result[0][x].args[0]

    return a, b, c


def slopeintercept_from_standard(a, b, c):

    x = 0
    a = float(a)
    b = float(b)
    c = float(c)
    y = -(a * x + c) / b
    # arctan2() is the quadrant correct version of arctan()
    angle = np.degrees(np.arctan2(-a, b))
    point = [x, y]
    return angle, point


def normal_from_standard(a, b, c):
    """Transfer a line's standard formulation (ax + by + c = 0) to normal.

    Args:
        a (float): a coef
        b (float): b coef
        c (float): c coef

    Returns:
        alpha (float): the angle between x axis and normal vector of the line, in degrees
        delta (float): the distance between zero point and the line
        
    https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Primka_rovnice_normalova.svg/1024px-Primka_rovnice_normalova.svg.png
    """
    c *= -1
    alpha = math.acos(a / (math.sqrt(math.pow(a, 2) + math.pow(b, 2))))
    delta = c / (math.sqrt(math.pow(a, 2) + math.pow(b, 2)))
    return math.degrees(alpha) + 90, delta


def normal_from_slopeintercept(angle, point):
    """Transfer a line's slope-intercept formulation to normal.

    Args:
        angle (float): the angle between x axis and the line, in degrees
        point (list [x coordinate, y coordinate]): any point on the line

    Returns:
        alpha (float): the angle between x axis and normal vector of the line, in degrees
        delta (float): the distance between zero point and the line
        
        https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Primka_rovnice_normalova.svg/1024px-Primka_rovnice_normalova.svg.png
    """
    a, b, c = standard_from_slopeintercept(angle, point)
    alpha, delta = normal_from_standard(a, b, c)
    alpha = angle + 90
    if alpha > 360:
        alpha -= 360
    return alpha, delta


def linesplit(alpha, delta, imshape, point_in_positive_halfplane=None):
    """Split image with a line in normal formulation. Use this with the plt.contour() function.

    Args:
        alpha (float): the angle between x axis and normal vector of the line, in degrees
        delta (float): the distance between zero point and the line
        imshape (int): square shape - one number, f.e. 512

    Returns:
        Linesplitted image
    """
    imshape = [imshape, imshape]
    orientation = alpha - 90
    x = math.cos(np.radians(alpha)) * delta
    y = math.sin(np.radians(alpha)) * delta
    logger.debug(f"x={x}, y={y}")
    point = [x, y]
    return bodynavigation.body_navigation.split_with_line(
        point,
        orientation,
        imshape,
        point_in_positive_halfplane=point_in_positive_halfplane,
    )

    # -5 -4 -3 -2 -1 0 1 2 3
    # -4 -3 -2


# a, b, c = standard_from_slopeintercept(2, [11, 5])
# print(f"Standard form: {a}x + {b}y + {c} = 0")


# alpha, delta = normal_from_slopeintercept(2, [11, 14])
# print(f"Normal form: alpha: {alpha}, delta: {delta}")
