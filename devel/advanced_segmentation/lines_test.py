from matplotlib import pyplot as plt
import bodynavigation.body_navigation as bona
from loguru import logger
import itertools
import numpy as np

import pytest
import lines

def test_angle():
    """
    Try combinations of angle and point to calculate a,b,c. Then the reverse calculation is applied.
    :return:
    """
    # angle = 0
    # point = [50,50]
    imsh = [100, 100]
    for angle, point in itertools.product(
        [45, -500, -360, -300, -180, -90, -30, -5, 0, 1,5,30,90,180, 200, 359,360,361, 500], # angles
        [[1, 1], [0,0], [50,50], [-100, 100], [-100, -50], [50, -50], np.asarray([0,0])] # points
    ):

        # im = bona.split_with_line(point, angle, imsh)
        # plt.imshow(im>0)
        # plt.show()

        logger.debug(f"input  angle: {angle}, point: {point}")
        a, b, c = lines.standard_from_slopeintercept(angle, point)
        angle1, point1 = lines.slopeintercept_from_standard(a,b,c)
        logger.debug(f"output angle: {angle1}, point: {point1}")
        logger.debug(f'a={a}, b={b}, c={c}')
        # assert a==0
        # assert b==0
        # assert c==0
        assert pytest.approx(angle, 5) == angle1
        assert pytest.approx(0, 5) == a * point[0] + b * point[1] + c

# lines.standard_from_slopeintercept(40,[1,1]) # nefunguje už ani tohle

test_angle()