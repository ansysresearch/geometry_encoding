from data.sympy_helper import *
from sympy import Symbol, Array
from sympy.utilities.lambdify import lambdify

"""
This files contains the implementation for the signed distance function of several 
geometries including circle, rectangle, triangle, diamond and cross-shape. 

The equation for the signed distance functions are obtained from 
 https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
 
All object are by default centered at origin. One may use .translate, .rotate or .resize 
methods to modify the objects. 
"""
#TODO: Triangle and IsosclesTriangle classes don't yet work properly.

class Geom:
    """
    The parent class which takes a parameters as params.
    The children class only need to specify the function compute_sdf.
    Assumed 2D geometry for now.
    """
    def __init__(self, params, compute_sdf=True):
        self.x = Symbol('x')
        self.y = Symbol('y')
        self.params = params
        self.sdf = None
        if compute_sdf: self.compute_sdf()

    def copy(self):
        cls = self.__class__
        new_copy = cls(self.params, compute_sdf=False)
        new_copy.sdf = self.sdf
        return new_copy

    # symbolic approach to compute sdf, will be populated in the children class.
    def compute_sdf(self):
        raise(NotImplementedError("Parent class does not have compute_sdf method."))

    # evaluate signed distance function at point x and y.
    # x and y can be 2d numpy arrays.
    # for this to work properly, we need a dictionary that specifies mapping
    # from sympy functions to numpy functions.
    def eval_sdf(self, x, y):
        sdf_lambdify = lambdify([self.x, self.y], self.sdf, modules=SIMPY_2_NUMPY_DICT)
        return sdf_lambdify(x, y)

    # plot binary and sdf images.
    def plot_sdf(self, x, y, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False):
        sdf = self.eval_sdf(x, y)
        img = sdf < 0
        plot_sdf(img, sdf, xticks=xticks, yticks=yticks, plot_eikonal=plot_eikonal)


    # move in the direction of the vector t[0], t[1]
    def translate(self, t):
        sub_dict = {self.x: self.x - t[0], self.y: self.y - t[1]}
        self.sdf = self.sdf.subs(sub_dict)
        return self

    # rotate with angle th (in radian) counter-clockwise
    # if we change both coordinate at the same time, something weird happens.
    # the second coordinate gets values of already changed first coordinate.
    def rotate(self, th):
        w = Symbol('w')
        sub_dict_1 = {self.x: np.cos(th) * self.x + np.sin(th) * w}
        self.sdf = self.sdf.subs(sub_dict_1)
        sub_dict_2 = {self.y: -np.sin(th) * self.x + np.cos(th) * self.y}
        self.sdf = self.sdf.subs(sub_dict_2)
        sub_dict_3 = {w: self.y}
        self.sdf = self.sdf.subs(sub_dict_3)
        return self

    # scale by s in x and y directions.
    # use the trick http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
    def scale(self, s):
        sub_dict = {self.x: self.x / s, self.y: self.y / s}
        self.sdf = self.sdf.subs(sub_dict) * s
        return self


class Circle(Geom):
    """
    circle geometry.
    params is a single number that is radius. (centered by default at origin).
    """
    def __repr__(self):
        return "circle of radius %0.2f" %self.params

    def compute_sdf(self):
        d = sympy_norm([self.x, self.y])
        self.sdf = d - self.params


class Rectangle(Geom):
    """
    rectangle geometry.
    params is a tuple of width and height of the rectangle (centered by default at origin).
    """
    def __repr__(self):
        return "rectangle of size (%0.2f, %0.2f)" %self.params

    def compute_sdf(self):
        qx, qy = abs(self.x) - self.params[0], abs(self.y) - self.params[1]
        self.sdf = sympy_norm([Max(qx, 0), Max(qy, 0)]) + Min(Max(qx, qy), 0)


class Diamond(Geom):
    """
    diamond geometry.
    params is a tuple of major and minor diagonals. (centered by default at origin).
    """
    def __repr__(self):
        return "diamond of size (%0.2f, %0.2f)" %self.params

    def compute_sdf(self):
        x, y = abs(self.x), abs(self.y)
        params = self.params
        h = (-2 * sympy_ndot([x, y], params) + sympy_ndot(params, params)) / sympy_dot(params, params)
        h = sympy_clamp(h, -1, 1)
        d = sympy_norm([x - 0.5 * params[0] * (1 - h),
                        y - 0.5 * params[1] * (1 + h)])
        sgn = f_sign(x * params[1] + y * params[0] - params[0] * params[1])
        self.sdf = d * sgn

class CrossX(Geom):
    """
    an X shape geometry.
    params is the tuple of width of the entire shape and width of each wing.
    """
    n_param = 2
    def compute_sdf(self):
        x = abs(self.x)
        y = abs(self.y)
        w, r = self.params
        d = Min(x + y, w)
        self.sdf = sympy_norm([x - 1/2 * d, y - 1/2 * d]) - r



class nGon(Geom):
    """
    a shape with n sides.
    params is a tuple of n coordinate pair of the vertices.
    it assumes the origin is inside the shape. Use .translate to move the shape if necessary.
    """
    #TODO: check shape include origins.

    def __repr__(self):
        return "Shape with %d sides" %len(self.params)

    def compute_sdf(self):

        # this function computes distance of point (x,y) from the line segment between (x1, y1) and (x2, y2)
        # see https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        def compute_distance_from_line(x, y, x1, y1, x2, y2):
            A = x - x1
            B = y - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D
            param = Max(dot / len_sq, 0)
            sgn1 = f_heaviside(param)
            sgn2 = f_heaviside(1 - param)
            xx = x1 + C * param * sgn1 * sgn2 + (1 - sgn2) * (x2 - x1)
            yy = y1 + D * param * sgn1 * sgn2 + (1 - sgn2) * (y2 - y1)
            dx = x - xx
            dy = y - yy
            return sqrt(dx * dx + dy * dy)

        # returns 1 for points inside shape and 0 otherwise.
        def get_sign():
            params = self.params.copy()
            params.append(params[0])
            sgn = 1
            for i in range(len(self.params)):
                x1, y1 = params[i]
                x2, y2 = params[i+1]
                if x1 == x2:
                    a = 1
                    b = 0
                    c = x1
                else:
                    a = (y2 - y1) / (x2 - x1)
                    b = -1
                    c = (x2 * y1 - x1 * y2) / (x2 - x1)
                l = a * self.x + b * self.y + c
                sgn *= f_heaviside(c*l)
            return sgn

        d = []
        params = self.params.copy()
        params.append(params[0])
        for i in range(len(self.params)):
            d.append(compute_distance_from_line(self.x, self.y, params[i][0], params[i][1],
                                                                params[i+1][0], params[i+1][1]))
        sgn1 = get_sign()
        self.sdf = f_min3(Array(d)) * (1 - 2 * sgn1)


def merge_geoms(geoms, x, y):
    sdf_pnts_vec = [geom.eval_sdf(x, y) for geom in geoms]
    sdf_pnts = np.minimum.reduce(sdf_pnts_vec)
    img_pnts = (sdf_pnts < 0).astype(float)
    return img_pnts, sdf_pnts


if __name__ == "__main__":
    EPS = 1e-2
    Nx, Ny = 1000, 1000
    x, y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    geoms = [#Circle(0.5), Circle(0.5).translate((0.5, 0.2)),
             Rectangle([0.2, 0.3]),
             Rectangle([0.2, 0.3]).translate((0.7, 0.7)),
             Rectangle([0.2, 0.3]).translate((0.7, 0.7)),

             Diamond([0.2, 0.3]).scale(1.5),
             nGon([[-0.3, -0.3], [0.3, 0.], [0., 0.6]]),
             Rectangle([0.4, 0.3]).rotate(1),
             CrossX([0.5, 0.1]),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]).translate((0.4, 0.2)),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]).translate((0.4, 0.2)).rotate(0.5),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]).translate((0.4, 0.2)).scale(0.4)
             ]
    for g in geoms:
        g.plot_sdf(x, y, plot_eikonal=True)


    geom = Rectangle([0.2, 0.3])
    geom2 = geom.copy().rotate(0.3)
    geom.plot_sdf(x, y)
    geom2.plot_sdf(x, y)

