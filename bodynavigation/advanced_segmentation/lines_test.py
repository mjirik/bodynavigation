from matplotlib import pyplot as plt
import bodynavigation.body_navigation as bona
from loguru import logger
import itertools
import numpy as np

import pytest
from bodynavigation.advanced_segmentation import lines


@pytest.mark.skip(reason="test is ok but there are problems in implemented")
def test_angle():
    """
    Try combinations of angle and point to calculate a,b,c. Then the reverse calculation is applied.
    :return:
    """
    # angle = 0
    # point = [50,50]
    imsh = [100, 100]
    for angle, point in itertools.product(
        [
            45,
            -500,
            -360,
            -300,
            -180,
            -90,
            -30,
            -5,
            0,
            1,
            5,
            30,
            90,
            180,
            200,
            359,
            360,
            361,
            500,
        ],  # angles
        [
            [1, 1],
            [0, 0],
            [50, 50],
            [-100, 100],
            [-100, -50],
            [50, -50],
            np.asarray([0, 0]),
        ],  # points
    ):

        # im = bona.split_with_line(point, angle, imsh)
        # plt.imshow(im>0)
        # plt.show()

        logger.debug(f"input  angle: {angle}, point: {point}")
        a, b, c = lines.standard_from_slopeintercept(angle, point)
        angle1, point1 = lines.slopeintercept_from_standard(a, b, c)
        logger.debug(f"output angle: {angle1}, point: {point1}")
        logger.debug(f"a={a}, b={b}, c={c}")
        # assert a==0.5
        # assert b==0
        # assert c==0
        assert pytest.approx(angle, 5) == angle1
        assert pytest.approx(0, 5) == a * point[0] + b * point[1] + c
        assert pytest.approx(0, 5) == a * point1[0] + b * point1[1] + c


# @pytest.mark.skip(reason="not implemented")
def test_linesplit():
    import bodynavigation.body_navigation

    angle = 30
    point = [10, 20]
    sh = [100, 100]
    alpha, delta = lines.normal_from_slopeintercept(angle, point)
    ls0 = bodynavigation.body_navigation.split_with_line(point, angle, sh)
    plt.subplot(221)
    plt.imshow(ls0)
    plt.contour(ls0 > 0)
    # plt.show()

    ls1 = lines.linesplit(alpha, delta, sh[0])
    logger.debug(ls1.shape)
    logger.debug(ls1.dtype)
    ls1 = ls1.astype(np.float)
    plt.subplot(222)
    plt.imshow(ls1)
    plt.contour(ls1 > 0)

    # try positive point
    ls2 = lines.linesplit(alpha, delta, sh[0], point_in_positive_halfplane=[60, 60])
    ls1 = ls2.astype(np.float)
    plt.subplot(223)
    plt.imshow(ls1)
    plt.contour(ls1 > 0)

    ls3 = lines.linesplit(
        alpha, delta, sh[0], point_in_positive_halfplane=np.asarray([10, 60])
    )
    ls1 = ls3.astype(np.float)
    plt.subplot(224)
    plt.imshow(ls1)
    plt.contour(ls1 > 0)

    # plt.show()

    assert ls2[0, 0] == -ls3[0, 0]


# lines.standard_from_slopeintercept(40,[0,0]) # nefunguje už ani tohle

# test_angle()

# test_linesplit()
