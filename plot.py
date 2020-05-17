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


def plot_direction(ax, miller, marker='+', pos='center', offset=0.5):
    """
    Mark a direction given by Miller indices.

    :param ax: Axes to be plotted to.
    :param miller: 3-tuple containing the Miller indices.
    :param marker: Symbol to be plotted at direction.
        An empty string or None means no symbol to be plotted.
    :param pos: Position of annotation. Valid choices are those that can
        also be passed to legend() by the loc argument. An empty string or None
        means no annotation.
    :param offset: Offset of the annotation in units of the font size.
        Not applied if pos='center'.
    """
    dir = np.asarray(miller, dtype=float)
    dir /= np.linalg.norm(dir)
    theta = np.arccos(dir[2])
    phi = np.arctan2(dir[1], dir[0])
    r = stereographic_projection(theta)
    x, y = cartesian(r, phi)

    if marker:
        ax.plot((x,), (y,), 'k', marker=marker)

    if pos:
        text = '['
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'$\overline{' + str(-m) + r'}$'
        text += ']'

        if pos == 'center':
            va = 'center'
            ha = 'center'
            transform = ax.transData
        else:
            align = {'upper': 'bottom', 'lower': 'top', 'center': 'center',
                     'left': 'right', 'right': 'left'}
            vpos, hpos = pos.split()
            va = align[vpos]
            ha = align[hpos]
            fontsize = mpl.rcParams['font.size']
            shift = offset * fontsize
            dx = {'left': shift, 'center': 0, 'right': -shift}
            dy = {'bottom': shift, 'center': 0, 'top': -shift}
            dx = dx[ha]
            dy = dy[va]
            transform = mpl.transforms.offset_copy(ax.transData, fig,
                                                   dx, dy, units='points')

        ax.text(x, y, text, ha=ha, va=va, transform=transform)


def plot_plane(ax, miller, frame, linestyle='-', pos='center',
               offset=0):
    """
    Show the plane by drawing a line.

    :param ax:  Axes to be plotted to.
    :param miller: 3-tuple containing the Miller indices.
    :param frame: Frame object to which the line representing the plane is to
        be clipped.
    :param linestyle: Line style (if empty or None, no line is drawn)
    :param pos: Position of annotation. Valid choices are 'top',
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

    points = frame.intersect(plane)
    if len(points) != 2:
        print(points)
        print('plot_plane: skipping ' + str(miller))
        return

    if straight:
        xy = np.array(points)

        dirt = xy[1] - xy[0]
        dirt /= np.linalg.norm(dirt)
        dirn = np.array((-dirt[1], dirt[0]))
        text_center = 0.5 * (xy[0] + xy[1])
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
        print(np.degrees(anglen))
        dirn = np.array((np.cos(anglen), np.sin(anglen)))
        text_center = np.array((plane.xc + plane.r * dirn[0],
                                plane.yc + plane.r * dirn[1]))
        text_rotation = anglen - np.pi/2

    if linestyle:
        path = Polygon(xy, closed=False, fill=False, linestyle=linestyle)
        ax.add_patch(path)

    if pos:
        # text
        text = '($'
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'\overline{' + str(-m) + r'}'
        text += '$)'

        # offset
        fontsize = mpl.rcParams['font.size']
        shift = offset * fontsize
        delta = {'top': shift, 'center': 0, 'bottom': -shift}
        delta = delta[pos]
        dx = delta * dirn[0]
        dy = delta * dirn[1]
        va = {'top': 'bottom', 'center': 'center', 'bottom': 'top'}
        va = va[pos]

        transform = mpl.transforms.offset_copy(ax.transData, fig,
                                               dx, dy, units='points')
        ax.text(*text_center, text, ha='center', va=va, transform=transform,
                rotation=np.degrees(text_rotation))


if __name__ == '__main__':
    fig, ax = plt.subplots()
    frame = Wedge()
    frame_polygon = frame.get_polygon()
    frame_patch = Polygon(frame_polygon, closed=True, fill=False, color='k')

    ax.add_patch(frame_patch)

    plot_direction(ax, (2,1,3), pos='center right')
    plot_direction(ax, (1,0,1), pos='lower right', marker=None)
    plot_direction(ax, (1,1,1), pos='upper right', marker=None)
    plot_direction(ax, (0,0,1), pos='lower left', marker=None)

    plot_plane(ax, (1,-2,0), frame, '--', pos='top')
    plot_plane(ax, (1,1,-1), frame, '--', pos='bottom')

    ax.set_aspect('equal')
    plt.show()
