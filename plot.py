"""
Plot functions for channeling maps.
"""
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sympy.geometry as geom

from geom import Line, Circle
from stereographic import stereographic_projection, cartesian
from wedge import Wedge


def loc2align(loc):
    """
    Helper function to convert relative location to alignment.

    Location is the placement of the object relative to the anchor; Alignment
    is the position of the anchor relative to the object.

    :param loc: '<vloc> <hloc>', where <vloc> is one of 'bottom', 'center',
        'top', and <hloc> is one of 'left', 'center', 'right'. <hloc> may be
        omitted.
    :return (va, ha): vertical and horizontal alignment.
    """
    loc = loc.split()
    vloc = loc[0]
    if len(loc) > 1:
        hloc = loc[1]
    else:
        hloc = 'center'

    va = {'top': 'bottom', 'center': 'center', 'bottom': 'top'}
    va = va[vloc]
    ha = {'left': 'right', 'center': 'center', 'right': 'left'}
    ha = ha[hloc]

    return va, ha


def plot_text(text, position, ha='center', va='center', offset=0, rotation=0,
              fig=None, ax=None):
    """
    Plot text at given position with offset an rotation.

    :param text: Text to be plotted.
    :param position: 2-tuple specifying the (x,y) position.
    :param ha: horizontal alignment ('left', 'center', or 'right').
    :param va: vertical alignment ('top', 'center', or 'bottom').
    :param offset: offset of text anchor point from position both in text
        rotation direction and perpendicular to it in units of the font size.
    :param rotation: rotation anlge (deg) counterclockwise relative to
        horizontal.
    :param fig: Figure to be plotted to.
    :param ax: Axes to be plotted to.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    dirt = np.array((np.cos(np.radians(rotation)),
                     np.sin(np.radians(rotation))))
    dirn = np.array((- dirt[1], dirt[0]))

    # shift anchor point
    fontsize = mpl.rcParams['font.size']
    shift = offset * fontsize
    dh = {'left': shift, 'center': 0, 'right': -shift}
    dv = {'bottom': shift, 'center': 0, 'top': -shift}
    dh = dh[ha]
    dv = dv[va]
    shift = dh * dirt + dv * dirn
    dx, dy = shift
    transform = mpl.transforms.offset_copy(ax.transData, fig,
                                           dx, dy, units = 'points')

    # plot the text
    ax.text(*position, text, ha=ha, va=va, transform=transform,
            rotation=rotation)


def plot_direction(miller, marker='+', loc='center', offset=0.5,
                   fig=None, ax=None):
    """
    Mark a direction given by Miller indices.

    :param fig: Figure to be plotted to
    :param ax: Axes to be plotted to.
    :param miller: 3-tuple containing the Miller indices.
    :param marker: Symbol to be plotted at direction.
        An empty string or None means no symbol to be plotted.
    :param loc: Position of annotation. Valid choices are those that can
        also be passed to legend() by the loc argument. An empty string or None
        means no annotation.
    :param offset: Offset of the annotation in units of the font size.
        Not applied if pos='center'.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    dir = np.asarray(miller, dtype=float)
    dir /= np.linalg.norm(dir)
    theta = np.arccos(dir[2])
    phi = np.arctan2(dir[1], dir[0])
    r = stereographic_projection(theta)
    x, y = cartesian(r, phi)

    if marker:
        ax.plot((x,), (y,), 'k', marker=marker)

    if loc:
        text = '['
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'$\overline{' + str(-m) + r'}$'
        text += ']'

        va, ha = loc2align(loc)

        plot_text(text, (x, y), ha, va, offset)


def plot_plane(ax, miller, frame, linestyle='-', loc='center',
               offset=0):
    """
    Show the plane by drawing a line.

    :param ax:  Axes to be plotted to.
    :param miller: 3-tuple containing the Miller indices.
    :param frame: Frame object to which the line representing the plane is to
        be clipped.
    :param linestyle: Line style (if empty or None, no line is drawn)
    :param loc: Position of annotation. Valid choices are 'top',
        'center', and 'bottom'
    :param offset: Offset of the annotation in units of the font size.
        Not applied if pos='center' (if empty or None, no annotation is
        written).
    """
    normal = np.asarray(miller, dtype=float)
    normal /= np.linalg.norm(normal)
    straight = (normal[2] == 0)

    # parameters describing the plane
    if straight:
        phi = - np.arctan2(normal[0], normal[1])
        plane = Line(point1=(0,0), point2=(np.cos(phi),np.sin(phi)))
    else:
        r = - 2 / normal[2]
        phic = np.arctan2(normal[1], normal[0])
        rc = - r * np.hypot(normal[0], normal[1])
        xc, yc = cartesian(rc, phic)
        plane = Circle(xc, yc, r)

    # intersections with frame
    points = frame.intersect(plane)
    if len(points) != 2:
        print(points)
        print('plot_plane: skipping ' + str(miller))
        return

    # text position and rotation
    if straight:
        xy = np.array(points)

        dirt = xy[1] - xy[0]
        dirt /= np.linalg.norm(dirt)
        dirn = np.array((-dirt[1], dirt[0]))
        text_position = 0.5 * (xy[0] + xy[1])
        text_rotation = np.arctan2(dirt[1], dirt[0])
    else:
        angle1 = np.arctan2(points[0][1] - plane.yc,
                            points[0][0] - plane.xc)
        angle2 = np.arctan2(points[1][1] - plane.yc,
                            points[1][0] - plane.xc)
        # assume difference betwenn angles < 180 deg
        dangle = angle2 - angle1
        if -np.pi < dangle < 0 or dangle > np.pi:
            angle1, angle2 = angle2, angle1
        xy = plane.get_polygon(angle1, angle2)

        anglen = angle1 + 0.5*abs(dangle)       # bisector, needed for pos
        dirn = np.array((np.cos(anglen), np.sin(anglen)))
        text_position = np.array((plane.xc + plane.r * dirn[0],
                                  plane.yc + plane.r * dirn[1]))
        text_rotation = anglen - np.pi/2

    if linestyle:
        path = Polygon(xy, closed=False, fill=False, linestyle=linestyle)
        ax.add_patch(path)

    if loc:
        # text
        text = '($'
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'\overline{' + str(-m) + r'}'
        text += '$)'

        va, ha = loc2align(loc)
        text_rotation = np.degrees(text_rotation)

        plot_text(text, text_position, ha, va, offset, text_rotation)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    frame = Wedge()
    frame_polygon = frame.get_polygon()
    frame_patch = Polygon(frame_polygon, closed=True, fill=False, color='k')

    ax.add_patch(frame_patch)

    plot_direction((2,1,3), loc='center right')
    plot_direction((1,0,1), loc='bottom right', marker=None)
    plot_direction((1,1,1), loc='top right', marker=None)
    plot_direction((0,0,1), loc='bottom left', marker=None)

    plot_plane(ax, (1,-2,0), frame, '--', loc='top')
    plot_plane(ax, (1,1,-1), frame, '--', loc='bottom')

    ax.set_aspect('equal')
    plt.show()
