import plyfile
import skimage.measure
from data.sympy_helper import *
from data.utils import plot_ply
from data.geoms import Geom
from sympy import Symbol, Array
from sympy.utilities.lambdify import lambdify
from mpl_toolkits.mplot3d import Axes3D


class Geom3(Geom):
    """
    Extension of 2D Geom into 3D. See geoms.py

    The children classes contain analytical expression for the signed distance function of several
    3D geometries including TODO fill here.

    The equation for the signed distance functions are obtained from
    https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

    All object are by default centered at origin. One may use .translate, .rotate or .resize
    methods to modify the objects.
    """
    def __init__(self, params, compute_sdf=True):
        self.z = Symbol('z', real=True)
        super().__init__(params, compute_sdf=compute_sdf)

    def lambdified_sdf(self):
        """
        evaluate signed distance function at point x and y.
        x and y can be 2d numpy arrays.
        for this to work properly, we need a dictionary that specifies mapping
        from sympy functions to numpy functions.

        :return: function f, where sdf = f(x,y,z)
        """
        return lambdify([self.x, self.y, self.z], self.sdf, modules=SIMPY_2_NUMPY_DICT)

    def eval_sdf(self, x, y, z=None):
        """
        evaluate sign distance function at point (x,y, [z]) .
        ** we need to introduce z to keep the same function signature for the children class Geom3.

        :param x: x-coordinate, np.array
        :param y: y-coordinate, np.array
        :param z: z-coordinate. np.array
        :return: sdf value, np.array
        """
        if z is None:
            raise(ValueError("z should be loaded for 3d geometry."))
        sdf_lambdify = self.lambdified_sdf()
        return sdf_lambdify(x, y, z)

    def translate(self, t):
        """
        move in the direction of the vector t[0], t[1], t[2]

        :param t: translation vector
        :return: translated object.
        """
        sub_dict = {self.x: self.x - t[0], self.y: self.y - t[1], self.z: self.z - t[2]}
        self.sdf = self.sdf.subs(sub_dict)
        return self

    def rotate(self, th, plane="xy"):
        """
        rotate with angle th (in radian) in plane xy
        ** if we change both coordinate at the same time, something weird happens.
        ** the second coordinate gets values of already changed first coordinate.
        ** hence w is introduced

        :param th: angle of rotation
        :param plane: plane of ration
        :return: rotated object
        """
        if plane == "xy":
            w = Symbol('w')
            sub_dict_1 = {self.x: np.cos(th) * self.x + np.sin(th) * w}
            self.sdf = self.sdf.subs(sub_dict_1)
            sub_dict_2 = {self.y: -np.sin(th) * self.x + np.cos(th) * self.y}
            self.sdf = self.sdf.subs(sub_dict_2)
            sub_dict_3 = {w: self.y}
            self.sdf = self.sdf.subs(sub_dict_3)
        else:
            raise(NotImplementedError("Other rotation planes are yet to be implemented"))
        return self

    def scale(self, s):
        """
        scale by s in x, y and z directions.
        use the trick http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/

        :param s: scaling factor
        :return: scaled object
        """
        sub_dict = {self.x: self.x / s, self.y: self.y / s, self.z: self.z / s}
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
            sub_dict = {self.z: self.z - Min(Max(self.z, -h), h)}
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

    def plot_sdf(self, x, y, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False):
        raise(NameError(
            """This function is for 2d geometries \n
               use plot_sdf_3d or plot_3d_from_sdf for 3d geometries."""
        ))

    def plot_sdf_3d(self, x, y, z):
        x_, y_, z_ = np.meshgrid(x, y, z)
        sdf = self.eval_sdf(x_, y_, z_)
        Axes3D.plot_trisurf(x, y, z, c=sdf)
        raise(NotImplementedError("Not yet implemented."))

    def plot_3d_from_sdf(self, x, y, z):
        """
        Convert sdf samples to .ply file, taken from Siren github
        https://github.com/vsitzmann/siren/blob/master/sdf_meshing.py

        :param x: x-coordinate, np.array
        :param y: y-coordinate, np.array
        :param z: z-coordinate. np.array
        """
        assert len(x) == len(y) == len(z), "only uniform voxels are supported right now."
        sdf = self.eval_sdf(x, y, z)
        assert np.any(sdf > 0), "sdf is all -ve, no objects found"
        assert np.any(sdf < 0), "sdf is all +ve, no objects found"

        voxel_size = 2 / len(x)
        voxel_grid_origin = (-1, -1, -1)

        verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(sdf, level=0.0, spacing=[voxel_size] * 3)
        except:
            raise(BrokenPipeError("marching_cubes did not work."))

        # transform from voxel coordinates to camera coordinates
        # note x and y are flipped in the output of marching_cubes
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

        # try writing to the ply file
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        plot_ply(ply_data)


class Sphere(Geom3):
    """
    geometry geometry.
    params is a single number that is radius. (centered by default at origin).
    """
    def __repr__(self):
        return "sphere of radius %0.2f" % self.params

    def compute_sdf(self):
        d = sympy_norm_3d([self.x, self.y, self.z])
        self.sdf = d - self.params


class Ellipsoid(Geom3):
    """
    ellipsoid geometry.
    params includes three radii. (centered by default at origin).
    """

    def __repr__(self):
        return "ellipsoid of of radii (%0.2f, %0.2f, %0.2f)" % self.params

    def compute_sdf(self):
        x, y, z = self.x, self.y, self.z
        rx, ry, rz = self.params
        k1 = sympy_norm_3d([x / rx, y / ry, z / rz])
        k2 = sympy_norm_3d([x / rx ** 2, y / ry ** 2, z / rz ** 2])
        self.sdf = k1 * (k1 - 1) / k2


class Capsule(Geom3):
    def __repr__(self):
        return "capsule"

    def compute_sdf(self):
        ax, ay, az, bx, by, bz, r = self.params
        x, y, z = self.x, self.y, self.z
        pa = [x - ax, y - ay, z - az]
        ba = [bx - ax, by - ay, bz - az]
        h = sympy_clamp(sympy_dot_3d(pa, ba) / sympy_dot_3d(ba, ba), 0, 1)
        self.sdf = sympy_norm_3d([pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h]) - r


class Cylinder(Geom3):
    def __repr__(self):
        return "cylinder"

    def compute_sdf(self):
        h, r = self.params
        x, y, z = self.x, self.y, self.z
        dx = sympy_norm([x, z]) - h
        dy = abs(y) - r
        tmp1 = Max(dx, dy)
        tmp2 = Min(tmp1, 0)
        tmp3 = sympy_norm([Max(dx, 0), Max(dy, 0)])
        self.sdf = tmp2 + tmp3


class Box(Geom3):
    """
    rectangular box geometry.
    params is a tuple of width, height and depth of the rectangle (centered by default at origin).
    """

    def __repr__(self):
        return "rectangle of size (%0.2f, %0.2f, %0.2f)" % self.params

    def compute_sdf(self):
        w, h, d = self.params
        qx, qy, qz = abs(self.x) - w, abs(self.y) - h, abs(self.z) - d
        tmp = f_max3(Array([qx, qy, qz]))
        self.sdf = sympy_norm_3d([Max(qx, 0), Max(qy, 0), Max(qz, 0)]) + Min(tmp, 0)


class RoundedBox(Geom3):
    """
    rounded rectangular box geometry.
    params is a tuple of width, height and depth of the rectangle and r (rounding radius)
    (centered by default at origin)
    """

    def __repr__(self):
        return "rounded rectangle of size (%0.2f, %0.2f, %0.2f) with rounding radius %0.2f" % self.params

    def compute_sdf(self):
        w, h, d, r = self.params
        qx, qy, qz = abs(self.x) - w, abs(self.y) - h, abs(self.z) - d
        tmp = f_max3(Array([qx, qy, qz]))
        self.sdf = sympy_norm_3d([Max(qx, 0), Max(qy, 0), Max(qz, 0)]) + Min(tmp, 0) - r


class HollowBox(Geom3):
    """
    hollow rectangular box geometry.
    params is a tuple of width, height and depth of the rectangle and hole length e
    (centered by default at origin)
    """

    def __repr__(self):
        return "rounded rectangle of size (%0.2f, %0.2f, %0.2f) with hole length %0.2f" % self.params

    def compute_sdf(self):
        w, h, d, e = self.params
        x, y, z = abs(self.x) - w, abs(self.y) - h, abs(self.z) - d
        qx, qy, qz = abs(x + e) - e, abs(y + e) - e, abs(z + e) - e
        px, py, pz = Max(x, 0), Max(y, 0), Max(z, 0)
        qqx, qqy, qqz = Max(qx, 0), Max(qy, 0), Max(qz, 0)
        tmp1 = sympy_norm_3d([px, qqy, qqz])
        tmp2 = sympy_norm_3d([qqx, py, qqz])
        tmp3 = sympy_norm_3d([qqx, qqy, pz])
        tmp4 = f_max3(Array([x, qy, qz]))
        tmp5 = f_max3(Array([qx, y, qz]))
        tmp6 = f_max3(Array([qx, qy, z]))
        tmp7 = Min(tmp4, 0)
        tmp8 = Min(tmp5, 0)
        tmp9 = Min(tmp6, 0)
        tmp10 = Array([tmp1 + tmp7, tmp2 + tmp8, tmp3 + tmp9])
        self.sdf = f_min3(tmp10)


class Torus(Geom3):
    """
    torus geometry.
    params include
    """

    def __repr__(self):
        return "torus (%0.2f, %0.2f)" % self.params

    def compute_sdf(self):
        pxz = sympy_norm([self.x, self.z])
        qx, qy  = pxz - self.params[0], self.y
        self.sdf = sympy_norm([qx, qy]) - self.params[1]


class CappedTorus(Geom3):
    """
    capped torus geometry.
    params include
    ToDO: Does not work . !!!
    """

    def __repr__(self):
        return "caped torus (%0.2f, %0.2f)" % self.params

    def compute_sdf(self):
        x, y, z = abs(self.x), self.y, self.z
        scx, scy, ra, rb = self.params
        cond = scy * x - scx * y
        tmp1 = sympy_dot([x, y], [scx, scy])
        tmp2 = sympy_norm([x, y])
        tmp3 = f_heaviside(cond) * (tmp1 - tmp2) + tmp2
        tmp4 = sympy_norm_3d([x, y, z]) + ra ** 2 - 2 * ra * tmp3
        self.sdf = sqrt(tmp4) - rb


class CappedCone(Geom3):
    """
    capped cone
    """

    def __repr__(self):
        return "caped cone"

    def compute_sdf(self):
        x, y, z = self.x, self.y, self.z
        h, r1, r2 = self.params
        k1x, k1y = r2, h
        k2x, k2y = r2 - r1, 2 * h
        qx, qy = sympy_norm([x, z]), y
        cond = f_heaviside(-y)
        tmp1 = (r1 - r2) * cond + r2
        tmp2 = Min(qx, tmp1)
        cax, cay = qx - tmp2, abs(qy) - h
        tmp3 = sympy_clamp(sympy_dot([k1x - qx, k1y - qy], [k2x, k2y]) / sympy_norm([k2x, k2y]), 0, 1)
        cbx, cby = qx - k1x + k2x * tmp3, qy - k1y + k2y * tmp3
        cond2 = f_heaviside(-cbx)
        cond3 = f_heaviside(-cay)
        s = cond2 * cond3 * (-1 - 1) + 1
        self.sdf = s * sqrt(Min(sympy_norm([cax, cay]), sympy_norm([cbx, cby])))


class RectangularRing(Geom3):
    """
    rectangular ring geometry.
    params include length, radius1, and radius2
    TODO: DOES NOT WORK!!!
    """

    def __repr__(self):
        return "rectangular ring with (l, r1, r2)= (%0.2f, %0.2f, %0.2f)" % self.params

    def compute_sdf(self):
        x, y, z = self.x, self.y, self.z
        l, r1, r2 = self.params
        tmp1 = Max(abs(y) - l, 0)
        tmp2 = sympy_norm([x, tmp1]) - r1
        self.sdf = sympy_norm([tmp2, z]) - r2


class Octahedron(Geom3):
    """
    octahedron geometry.
    """

    def __repr__(self):
        return "octahedron"

    def compute_sdf(self):
        x, y, z = abs(self.x), abs(self.y), abs(self.z)
        s = self.params
        m = x + y + z - s
        cond1 = f_heaviside(-x + m / 3)
        cond2 = f_heaviside(-y + m / 3)
        cond3 = f_heaviside(-z + m / 3)
        qx = cond1 * x + (1-cond1) * (cond2 * y + (1 - cond2) * (cond3 * z))
        qy = cond1 * y + (1-cond1) * (cond2 * z + (1 - cond2) * (cond3 * x))
        qz = cond1 * z + (1-cond1) * (cond2 * x + (1 - cond2) * (cond3 * y))
        k = sympy_clamp(0.5 * (qz - qy + s), 0, s)
        v1 = m * 0.57735
        v2 = sympy_norm_3d([qx, qy - s + k, qz - k])
        cond4 = (1 - cond1) * (1 - cond2) * (1 - cond3)
        self.sdf = cond4 * v1 + (1 - cond4) * v2


class HexagonalPrism(Geom3):
    def __repr__(self):
        return "hexagonal prism with parameters (%0.2f, %0.2f)"%self.params

    # def compute_sdf(self):
    #     kx, ky, kz = -0.8660254, 0.5, 0.57735
    #     hx, hy = self.params
    #     x, y, z = abs(self.x), abs(self.y), abs(self.z)
    #     tmp1 = 2 * Min(sympy_dot([kx, ky], [x, y]), 0)
    #     px, py = x - tmp1 * kx, y - tmp1 * ky
    #     tmp2 = sympy_clamp(px, -kz * hx, kz * hx)
    #     tmp3 = sympy_norm([px - tmp2, py - hx])
    #     tmp4 = tmp3 * f_sign(y - hx)
    #     tmp5 = Max(tmp4, z - hy)
    #     tmp6 = Min(tmp5, 0)
    #     tmp7 = Max(tmp4, 0)
    #     tmp8 = Max(z - hy, 0)
    #     tmp9 = sympy_norm([tmp7, tmp8])
    #     self.sdf = tmp6 + tmp9

    def compute_sdf(self):
        x, y, z = self.x, self.y, self.z
        ax, ay, az, bx, by, bz, ra, rb = self.params
        rba = rb - ra
        baba = sympy_norm_3d([bx - ax, by - ay, bz - az])
        papa = sympy_norm_3d([x - ax, y - ay, z - az])
        paba = sympy_dot_3d([x - ax, y - ay, z - az]) / baba
        w = sqrt(papa - paba * paba * baba)
        cond1 = f_heaviside(0.5 - paba)
        cax = Max(w - cond1 * (ra - rb) + rb, 0)
        cay = abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = sympy_clamp((rba * (w - ra) + paba * baba) / k, 0.0, 1.0)
        cbx = w - ra - f * rba
        cby = paba - f
        cond2 = f_heaviside(-cbx)
        cond3 = f_heaviside(-cay)
        s = cond2 * cond3 * (-2) + 1
        self.sdf = s * sqrt(Min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))


if __name__ == "__main__":
    n = 128
    x_ = np.linspace(-1., 1., n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')

    # s = Box((0.4, 0.1, 0.3))
    # s.plot_3d_from_sdf(x, y, z)
    #
    # s = RoundedBox((0.25, 0.4, 0.1, 0.1))
    # s.plot_3d_from_sdf(x, y, z)
    #
    # t = Torus((0.5, 0.1))
    # t.plot_3d_from_sdf(x, y, z)
    # t.rotate(1).scale(2.0).plot_3d_from_sdf(x, y, z)
    #
    # ct = RectangularRing((0.5, 0.2, 0.1))
    # ct.plot_3d_from_sdf(x, y, z)
    #
    # hp = HexagonalPrism((0.4, 0.3))
    # hp.plot_3d_from_sdf(x, y, z)
    #
    # c = Capsule((0.1, -0.1, 0.1, 0.4, 0.4, 0.4, 0.1))
    # c.plot_3d_from_sdf(x, y, z)
    #
    # c = Cylinder((0.1, 0.7))
    # c.plot_3d_from_sdf(x, y, z)
    cc = CappedCone((0.2, 0.2, 0.1))
    cc.plot_3d_from_sdf(x, y, z)

    ct = CappedTorus((0.25, 0.15, 0.3, 0.5))
    ct.plot_3d_from_sdf(x, y, z)

    hb = HollowBox((0.5, 0.4, 0.3, 0.1))
    hb.plot_3d_from_sdf(x, y, z)

    oh = Octahedron(0.3)
    oh.plot_3d_from_sdf(x, y, z)

    e = Ellipsoid((0.3, 0.4, 0.2))
    e.plot_3d_from_sdf(x, y, z)
    e.elongate(0.5, along='z').plot_3d_from_sdf(x, y, z)

    s = Torus((0.5, 0.2))
    s.onion(0.25).plot_3d_from_sdf(x, y, z)
