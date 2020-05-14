"""
Collect functions related to the stereographic projection.

A stereographic projection is a mapping between a direction in 3D space and a
position in a 2D plane. The direction can be described in polar coordinates
by (theta,phi), where theta denotes the angle between the direction and
the z axis, and phi denotes the azimuthal angle. The point in the 2D
plane can be described by its Cartesian coordinates (x,y) or its polar
coordinates (r,phi). phi is the same for the direction and the point.
"""
import numpy as np


def stereographic_projection(theta, phi=None):
    """
    Perform the stereographic projection (theta,phi) -> (r,phi).

    :param theta: polar angle (rad)
    :param phi: azimuthal angle (rad)
    :return (r,phi): polar coordinates in the plane
    """
    r = 2 * np.tan(0.5*theta)

    if phi is None:
        return r
    else:
        return r, phi


def inverse_stereographic_projection(r, phi=None):
    """
    Perform the inverse stereographic projection (r,phi) -> (theta,phi).

    :param r: distance from origin
    :param phi: azimuthal angle (rad)
    :return (theta,phi): polar coordinates of the direction
    """
    theta = 2 * np.arctan(0.5*r)

    if phi is None:
        return theta
    else:
        return theta, phi


def cartesian(r, phi):
    """
    Convert polar to Cartesian coordinates.

    :param r: Distance from origin
    :param phi: azimutal angle
    :return (x,y): Cartesian coordinates
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def polar(x, y):
    """
    Convert Cartesian to polar coordinates.

    :param x: coordinate along the x axis
    :param y: coordinate along the y axis
    :return (r,phi): (radius, azimuthal angle)
    """
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)

    return r, phi


def rwedge(phi):
    """
    Calculate the radius r along the line connecting [011] and [111].

    :param phi: azimuthal angle (rad)
    :return r: radius
    """
    r = 2 * (np.sqrt(1 + np.cos(phi)**2) - np.cos(phi))

    return r


def get_wedge(dphi=1):
    """
    Calculate the wedge connecting the [001], [101], and [111] direction.

    The line from [001] over [101] to [111] is returned.

    :param dphi: approximate phi increment along the line [101]-[111] (deg)

    :return (x, y): (x coordinates, y coordinates)
    """
    wedge = []

    # line from [101] to [111]
    phis = np.linspace(0, 45, (45/dphi)+1, endpoint=True)
    phis = np.radians(phis)
    rs = rwedge(phis)
    xs, ys = cartesian(rs, phis)
    # [001]
    xs = np.insert(xs, 0, 0.)
    ys = np.insert(ys, 0, 0.)

    return xs, ys


def get_triangle(theta_max):
    """
    Get triangle (0,0)-(xmax,0)-(xmax,xmax), xmax corresponding to theta_max.

    :param theta_max: Maximum angle along x direction (deg)
    :return (x,y): (x coordinates, y coordinates)
    """
    xmax = stereographic_projection(np.radians(theta_max))
    xs = np.array((0., xmax, xmax))
    ys = np.array((0., 0., xmax))

    return xs, ys


# not sure where we need this:
def intersect_bounding_box(circle, bbox):
    xc, yc, r = circle
    xmin, ymin, xmax, ymax = bbox
    intersections = []
    # intersections with x=xmin and x=xmax
    for x in (xmin, xmax):
        discriminant = r**2 - (x-xc)**2
        if discriminant >= 0:
            for sign in (1, -1):
                y = yc + sign * np.sqrt(discriminant)
                if y >= ymin and y <= ymax:
                    intersections.append((x, y))
    # intersections with y=ymin and y=ymax
    for y in (ymin, ymax):
        discriminant = r**2 - (y-yc)**2
        if discriminant >= 0:
            for sign in (1, -1):
                x = xc + sign * np.sqrt(discriminant)
                if x >= xmin and x <= xmax:
                    intersections.append((x, y))
    # remove duplicates (could be in the corners)
    intersections = tuple(set(intersections))
    return intersections


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    xs, ys = get_wedge()
    plt.plot(xs, ys)
    xs, ys = get_triangle(10)
    plt.plot(xs, ys, '--')
    plt.gca().set_aspect('equal')
    plt.show()