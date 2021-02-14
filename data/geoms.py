from data.sympy_helper import *
from data.utils import plot_sdf
from sympy import Symbol, Array
from sympy.utilities.lambdify import lambdify
from scipy.ndimage import distance_transform_edt


class Geom:
    """
    The parent class which takes a parameters as params.
    The children class only need to specify the function compute_sdf.
    Assumed 2D geometry.  See geoms3.py for 3D geometry

    The children classes contain analytical expression for the signed distance function of several
    geometries including circle, rectangle, triangle, diamond and cross-shape.

    The equation for the signed distance functions are obtained from
    https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

    All object are by default centered at origin. One may use .translate, .rotate or .resize
    methods to modify the objects.
    """
    def __init__(self, params, compute_sdf=True):
        self.x = Symbol('x', real=True)
        self.y = Symbol('y', real=True)

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
    def lambdified_sdf(self):
        return lambdify([self.x, self.y], self.sdf, modules=SIMPY_2_NUMPY_DICT)

    def eval_sdf(self, x, y, z=None):
        """
        evaluate sign distance function at point (x,y, [z]) .
        ** we need to introduce z to keep the same function signature for the children class Geom3.

        :param x: x-coordinate, np.array
        :param y: y-coordinate, np.array
        :param z: z-coordinate. None for 2d geometries, or np.array for 3d geometries
        :return: sdf value, np.array
        """
        sdf_lambdify = self.lambdified_sdf()
        if z is not None:
            raise(ValueError("z should be loaded only for 3d geometry."))
        return sdf_lambdify(x, y)

    # plot binary and sdf images.
    def plot_sdf(self, x, y, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False):
        sdf = self.eval_sdf(x, y)
        img = sdf < 0
        plot_sdf(img, sdf, xticks=xticks, yticks=yticks, plot_eikonal=plot_eikonal)

    def translate(self, t):
        """
        move in the direction of the vector t[0], t[1]

        :param t: translation vector
        :return: translated object.
        """
        sub_dict = {self.x: self.x - t[0], self.y: self.y - t[1]}
        self.sdf = self.sdf.subs(sub_dict)
        return self

    # rotate with angle th (in radian) counter-clockwise
    # if we change both coordinate at the same time, something weird happens.
    # the second coordinate gets values of already changed first coordinate.
    def rotate(self, th):
        """
        rotate with angle th (in radian) counter-clockwise
        ** if we change both coordinate at the same time, something weird happens.
        ** the second coordinate gets values of already changed first coordinate.
        ** hence w is introduced

        :param th: angle of rotation
        :return: rotated object
        """
        w = Symbol('w')
        sub_dict_1 = {self.x: np.cos(th) * self.x + np.sin(th) * w}
        self.sdf = self.sdf.subs(sub_dict_1)
        sub_dict_2 = {self.y: -np.sin(th) * self.x + np.cos(th) * self.y}
        self.sdf = self.sdf.subs(sub_dict_2)
        sub_dict_3 = {w: self.y}
        self.sdf = self.sdf.subs(sub_dict_3)
        return self

    def scale(self, s):
        """
        scale by s in x and y directions.
        use the trick http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/

        :param s: scaling factor
        :return: scaled object
        """
        sub_dict = {self.x: self.x / s, self.y: self.y / s}
        self.sdf = self.sdf.subs(sub_dict) * s
        return self

    def elongate(self, h, along="x"):
        """
        elongate an object

        :param h: elongation factor
        :along : axis of alongation
        :return: elongated object
        """
        if along == "x":
            sub_dict = {self.x: self.x - Min(Max(self.x, -h), h)}
        elif along == "y":
            sub_dict = {self.y: self.y - Min(Max(self.y, -h), h)}
        elif along == "z":
            raise(ValueError("2D object does not have z dimension"))
        else:
            raise(ValueError("axis %s is not recognized, Axes are x, y, z. " % along))
        self.sdf = self.sdf.subs(sub_dict)
        return self

    def roundify(self, r):
        self.sdf -= r
        return self

    def onion(self, th):
        self.sdf = abs(self.sdf) - th
        return self

    def union(self, other):
        union_geom = self.copy()
        union_geom.sdf = Min(self.sdf, other.sdf)
        union_geom.params = [self, other]
        return union_geom


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


def compute_sdf_from_img1(img):
    assert np.ndim(img) == 2
    assert img.shape[0] == img.shape[1]
    sdf = np.zeros(img.shape)
    m, n = sdf.shape
    img_b_idx = np.array(np.where(img == 0)).T
    img_w_idx = np.array(np.where(img == 1)).T
    for i in range(m):
        for j in range(n):
            if img[i, j] == 1:
                sdf[i, j] = - np.min(np.sqrt(((img_b_idx - [i, j]) ** 2).sum(axis=1)))
            elif img[i, j] == 0:
                sdf[i, j] = np.min(np.sqrt(((img_w_idx - [i, j]) ** 2).sum(axis=1)))
            else:
                raise("ERROR")
    sdf = sdf / (img.shape[0] / 2)
    return sdf


def compute_sdf_from_img2(img):
    assert np.ndim(img) == 2
    assert img.shape[0] == img.shape[1]
    scipy_sdf = -distance_transform_edt(img) + distance_transform_edt(1 - img)
    scipy_sdf /= (img.shape[0] // 2)
    return scipy_sdf


def merge_geoms(geoms, x, y):
    sdf_pnts_vec = [geom.eval_sdf(x, y) for geom in geoms]
    sdf_pnts = np.minimum.reduce(sdf_pnts_vec)
    img_pnts = (sdf_pnts < 0).astype(float)
    return img_pnts, sdf_pnts


if __name__ == "__main__":
    EPS = 1e-2
    Nx, Ny = 200, 200
    x, y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    geoms = [Circle(0.5),
             Circle(0.5).translate((0.5, 0.2)).scale(0.3),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]).translate((0.4, .3)).rotate(1.),
             Rectangle([0.2, 0.3]),
             Rectangle([0.2, 0.3]).translate((0.7, 0.7)),
             Diamond([0.2, 0.3]).scale(1.5),
             CrossX([0.5, 0.1]),
             nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.5]]),
             ]
    # for g in geoms:
    #     g.plot_sdf(x, y, plot_eikonal=True)

    geom1 = Rectangle([0.5, 0.2])
    geom2 = nGon([[0.2, 0.2], [-0.5, 0.5], [-0.2, -0.8], [0.25, -0.75]])
    geom3 = geom2.copy().rotate(0.3).translate((0.3, 0.2))
    geoms = [geom1, geom2, geom3]
    for g in geoms:
        sdf = g.eval_sdf(x, y)
        img = sdf < 0
        plot_sdf(img, sdf, plot_eikonal=True, show=False)

    geom4 = geom1.union(geom3).union(geom2)
    sdf4 = geom4.eval_sdf(x, y)
    img4 = sdf4 < 0
    plot_sdf(img4, sdf4, plot_eikonal=True, show=True)