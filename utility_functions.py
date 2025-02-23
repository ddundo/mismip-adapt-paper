import os

import numpy as np
from numpy.linalg import eig
from animate.metric import RiemannianMetric
from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
from goalie import MeshSeq

from h5py import File as h5File
from icepack.constants import (
    gravity as g,
)
from icepack.constants import (
    ice_density as rho_I,
)
from icepack.constants import (
    water_density as rho_W,
)
from icepack.constants import (
    weertman_sliding_law as m,
)

from matplotlib.pyplot import tricontour


@PETSc.Log.EventDecorator()
def tau_metric(h, u, mp):
    mesh = h.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    Q = h.function_space()
    V = u.function_space()
    z_b = Function(Q).interpolate(mismip_bed_topography(mesh))
    s = Function(Q).interpolate(surface_expression(h, z_b))
    friction = friction_law(
        thickness=h, velocity=u, surface=s, friction=1e-2
    )
    basal_stress = Function(V).interpolate(-diff(friction, u))

    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)
    temp_metric = metric.copy(deepcopy=True)

    for i, m in enumerate([metric, temp_metric]):
        # bstress = Function(Q).interpolate(basal_stress[i])
        m.compute_hessian(basal_stress[i])
        # m.compute_hessian(bstress, method="Clement")
        # m.compute_hessian(bstress)
        m.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
        # m.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    metric.intersect(temp_metric)

    return metric

@PETSc.Log.EventDecorator()
def h_metric(h, u, mp):
    mesh = h.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)
    metric.compute_hessian(h)
    return metric

@PETSc.Log.EventDecorator()
def s_metric(h, u, mp):
    mesh = h.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)
    z_b = Function(h.function_space()).interpolate(mismip_bed_topography(mesh))
    s = Function(h.function_space()).interpolate(surface_expression(h, z_b))
    metric.compute_hessian(s)
    return metric

@PETSc.Log.EventDecorator()
def u_metric(h, u, mp):
    mesh = u.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)
    temp_metric = metric.copy(deepcopy=True)
    for i, m in enumerate([metric, temp_metric]):
        m.compute_hessian(u[i])
        m.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
        # m.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    metric.intersect(temp_metric)
    return metric

@PETSc.Log.EventDecorator()
def utau_metric(h, u, mp):
    hessian_u = u_metric(h, u, mp)
    hessian_tau = tau_metric(h, u, mp)
    for h in [hessian_u, hessian_tau]:
        h.normalise()
    metric = hessian_u.copy(deepcopy=True).intersect(hessian_tau)
    return metric

@PETSc.Log.EventDecorator()
def uh_metric(h, u, mp):
    hessian_u = u_metric(h, u, mp)
    hessian_h = h_metric(h, u, mp)
    for h in [hessian_u, hessian_h]:
        h.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    metric = hessian_u.copy(deepcopy=True).intersect(hessian_h)
    return metric

@PETSc.Log.EventDecorator()
def us_metric(h, u, mp):
    hessian_u = u_metric(h, u, mp)
    hessian_s = s_metric(h, u, mp)
    for h in [hessian_u, hessian_s]:
        h.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    metric = hessian_u.copy(deepcopy=True).intersect(hessian_s)
    return metric

@PETSc.Log.EventDecorator()
def isotropic_dwr_metric(h_ind, u_ind, mp):
    mesh = h_ind.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)
    temp_metric = metric.copy(deepcopy=True)
    for m, ind in zip([metric, temp_metric], [h_ind, u_ind]):
        m.compute_isotropic_dwr_metric(ind)
        m.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    metric.intersect(temp_metric)
    return metric

@PETSc.Log.EventDecorator()
def anisotropic_dwr_metric(h, u, mp):
    pass

@PETSc.Log.EventDecorator()
def anisotropic_weighted_hessian_metric(h_ind, u_ind, h, u, mp):
    mesh = h_ind.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(mp)

    hessian_h = h_metric(h, u, mp)
    hessian_u = u_metric(h, u, mp)

    metric.compute_weighted_hessian_metric([h_ind, u_ind], [hessian_h, hessian_u])
    return metric

def get_flux_and_melt(msh, h, u, options):
    Q = FunctionSpace(msh, "CG", 1)
    R = FunctionSpace(msh, "R", 0)
    v = FacetNormal(msh)

    flux = assemble(h * inner(u, v) * ds(tuple([2])))

    z_b = Function(Q).interpolate(mismip_bed_topography(msh))
    s = Function(Q).interpolate(surface_expression(h, z_b))
    omega = Function(R).assign(options["omega"])
    z_0 = Function(R).assign(options["z_0"])
    h_c0 = Function(R).assign(options["h_c0"])
    z_d = s - h
    h_c = z_d - z_b
    melt = omega * tanh(h_c / h_c0) * max_value(z_0 - z_d, 0.0)
    melt = assemble(melt * dx)

    return flux, melt

def analyse_meshseq(mesh_seq):
    output_dir = os.path.dirname(mesh_seq.output_fpath)
    analysis_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    def np_fpath(var_name): return os.path.join(analysis_dir, f"{var_name}.npy")

    tp = mesh_seq.time_partition
    sols = mesh_seq.solutions.extract(layout="label")["forward"]
    h_sols = np.array(sols["h"]).flatten()
    masses = get_mass(h_sols)
    hafs = get_haf(h_sols)
    vafs = get_vaf(hafs)
    gls = get_gl(hafs)
    midline_x_gls = get_midline_x_gl(gls)

    var_names = ["masses", "vafs", "gls", "midline_x_gls"]
    variables = [masses, vafs, gls, midline_x_gls]
    for var_name, variable in zip(var_names, variables):
        if os.path.exists(np_fpath(var_name)):
            var_fpath = np_fpath(var_name)
            load_var = np.load(var_fpath)
            load_var.concat(variable, axis=1)
            np.save(var_fpath, load_var)

def vtk_meshseq(msq, metrics, iteration):
    parent_class = msq.__class__.__bases__[0]
    tp = msq.time_partition
    solutions = msq.solutions.extract()
    if parent_class is not MeshSeq:
        indicators = msq.indicators.extract()

    for i in range(len(msq)):
        for j in range(tp.num_exports_per_subinterval[i] - 1):
            time = float(msq.get_time(i)) + j * tp.timesteps[i] * tp.num_timesteps_per_export[i]
            print(i, j, time)
            outfile = VTKFile(f"output_iter_{iteration}_int_{i}_yr_{time:.2f}.pvd")
            u_forward = solutions["u"]["forward"][i][j]
            h_forward = solutions["h"]["forward"][i][j]
            u_forward.rename(f"u_iter_{iteration}_int_{i}_t_{time}")
            h_forward.rename(f"h_iter_{iteration}_int_{i}_t_{time}")
            if parent_class is not MeshSeq:
                u_ind = indicators["u"][i][j]
                h_ind = indicators["h"][i][j]
                u_ind.rename(f"u_indicator_iter_{iteration}_int_{i}_t_{time}")
                h_ind.rename(f"h_indicator_iter_{iteration}_int_{i}_t_{time}")
                outfile.write(u_ind, h_ind, u_forward, h_forward, time=time)
            else:
                outfile.write(u_forward, h_forward, time=time)
        if metrics is not None:
            metric = metrics[i]
            metric.rename(f"metric_iter_{iteration}_int_{i}",)
            outfile.write(metric)

def checkpoint_meshseq(
        mesh_seq, save_sols=False, save_inds=False, iteration=None, metrics=None, k=1
        ):
    tp = mesh_seq.time_partition
    subinterval_length = tp.end_time / tp.num_subintervals
    num_exports_per_subinterval = int(tp.num_exports_per_subinterval[0] - 1)
    exports_per_year = int(num_exports_per_subinterval / subinterval_length)

    solutions = mesh_seq.solutions.extract()
    if save_inds:
        try:
            indicators = mesh_seq.indicators.extract()
        except TypeError as e:
            if "'NoneType' object is not subscriptable" in str(e):
                print("No indicators found. Skipping.")
                save_inds = False
            else:
                raise e

    # mesh name is mesh_iter_0_int_0
    name_iter = int(mesh_seq[0].name.split("mesh_iter_")[1].split("_")[0])
    iteration = iteration or name_iter

    print(f"Checkpointing iteration {iteration}.")

    with CheckpointFile(mesh_seq.output_fpath, "a") as afile:
        for i in range(len(solutions.u.forward)):
            afile.save_mesh(mesh_seq[i])
            for j in range(exports_per_year-1, num_exports_per_subinterval, exports_per_year):
                year = (i * subinterval_length + j + 1)
                if save_sols:
                    afile.save_function(
                        solutions.u.forward[i][j],
                        name=f"velocity_iter_{iteration}_int_{i}_yr_{year}",
                    )
                    afile.save_function(
                        solutions.h.forward[i][j],
                        name=f"thickness_iter_{iteration}_int_{i}_yr_{year}",
                    )
                if save_inds:
                    afile.save_function(
                        indicators.u[i][j],
                        name=f"velocity_ind_iter_{iteration}_int_{i}_yr_{year}",
                    )
                    afile.save_function(
                        indicators.h[i][j],
                        name=f"thickness_ind_iter_{iteration}_int_{i}_yr_{year}",
                    )

        if metrics is not None:
            for i, metric in enumerate(metrics):
                afile.save_function(metric, name=f"metric_iter_{iteration}_int_{i}")

    afile.close()

def surface_expression(h, b):
    return max_value(h + b, (1 - rho_I / rho_W) * h)


def tanh(z):
    return (exp(z) - exp(-z)) / (exp(z) + exp(-z))


def mismip_bed_topography(msh):
    x, y = SpatialCoordinate(msh)

    Ly = 80e3

    x_c = Constant(300e3)
    X = x / x_c

    B_0 = Constant(-150)
    B_2 = Constant(-728.8)
    B_4 = Constant(343.91)
    B_6 = Constant(-50.57)
    B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6

    f_c = Constant(4e3)
    d_c = Constant(500)
    w_c = Constant(24e3)

    B_y = d_c * (
        1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c))
        + 1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
    )

    z_deep = Constant(-720)
    return max_value(B_x + B_y, z_deep)


def friction_law(**kwargs):
    variables = ("velocity", "thickness", "surface", "friction")
    u, h, s, C = map(kwargs.get, variables)

    p_W = rho_W * g * max_value(0, -(s - h))
    p_I = rho_I * g * h
    N = max_value(0, p_I - p_W)
    tau_c = N / 2

    u_c = (tau_c / C) ** m
    u_b = sqrt(inner(u, u))

    return tau_c * ((u_c ** (1 / m + 1) + u_b ** (1 / m + 1)) ** (m / (m + 1)) - u_c)

def get_num_iterations(fpath):
    max_iter = 20
    num_subintervals = int(fpath.split("-id_")[-1].split("_")[2].split(".h5")[0])

    iterations = []
    with h5File(fpath, "r") as f:
        for iter in range(max_iter):
            mesh_name = f"mesh_iter_{iter}_int_{num_subintervals-1}"
            if f"{mesh_name}_topology" in f["topologies"].keys():
                try:
                    thickness_fns = list(
                        f[
                            f"topologies/{mesh_name}_topology/firedrake_meshes/{mesh_name}/firedrake_function_spaces/firedrake_function_space_{mesh_name}_CG1(None,None)/firedrake_functions/"
                        ]
                    )
                    thickness_fn_name = (
                        f"thickness_iter_{iter}_int_{num_subintervals-1}_yr_200.0"
                    )
                    if thickness_fn_name in thickness_fns:
                        iterations.append(iter)
                except KeyError:
                    break

    return iterations


def get_mass(hs, half_domain=True):
    if not isinstance(hs, (list, np.ndarray)):
        hs = [hs]
    factor = 2 if half_domain else 1

    masses = []
    for h in hs:
        masses.append(assemble(h * dx) / 1e9 / 917 * factor)  # in Gt w.e.

    return masses if len(masses) > 1 else masses[0]


def get_haf(hs):
    if not isinstance(hs, (list, np.ndarray)):
        hs = [hs]

    hafs = []
    for h in hs:
        fspace = h.function_space()
        z_b = assemble(interpolate(mismip_bed_topography(fspace.mesh()), fspace))
        s = assemble(interpolate(surface_expression(h, z_b), fspace))
        haf = assemble(interpolate(max_value(s - (1 - rho_I / rho_W) * h, 0), fspace))
        hafs.append(haf)

    return hafs if len(hafs) > 1 else hafs[0]

def get_vaf(hafs):
    if not isinstance(hafs, (list, np.ndarray)):
        hafs = [hafs]

    vafs = []
    for haf in hafs:
        vaf = assemble(haf * dx) / 1e9  # km^3
        vafs.append(vaf)

    return vafs if len(vafs) > 1 else vafs[0]


def get_gl(hafs):
    if not isinstance(hafs, list):
        hafs = [hafs]

    gls = []
    for haf in hafs:
        points = haf.function_space().mesh().coordinates.dat.data
        cs = tricontour(points[:, 0], points[:, 1], haf.dat.data, levels=[0.1])  # 0.01
        longest_sublist = max(cs.allsegs, key=len)[0]
        gls.append(longest_sublist)
        # gl = np.concatenate(cs.allsegs[0])
        # gls.append(gl)

    return gls if len(gls) > 1 else gls[0]

def tidy_gl(arr):
    # make a deep copy of the array arr
    arr = arr.copy()
    # remove nans
    arr = arr[~np.isnan(arr).any(axis=1)]
    # remove rows that repeat
    diffs = np.diff(arr, axis=0) != 0
    unique_rows_mask = np.insert(np.any(diffs, axis=1), 0, True)
    arr = arr[unique_rows_mask]

    # We'll store the cleaned data in a new list
    cleaned = []
    # We'll keep track of points we've seen and where we saw them
    seen_points = {}
    for i, point in enumerate(arr):
        # Convert point to tuple so it can be used as a dictionary key
        point_as_tuple = tuple(point)
        if point_as_tuple in seen_points:
            # If we've seen this point before, remove all points in-between
            first_occurrence_index = seen_points[point_as_tuple]
            cleaned = cleaned[:first_occurrence_index+1]
        # Add the point to the cleaned list and mark it as seen
        cleaned.append(point)
        seen_points[point_as_tuple] = len(cleaned) - 1
    return np.array(cleaned)

def get_midline_x_gl(gls, y0=4e4):
    midline_x_gls = []

    for gl in gls:
        x_gl = gl[:, 0]
        y_gl = gl[:, 1]
        midline_x_gl = x_gl[np.where(y_gl == y0)]
        if len(midline_x_gl) == 0:
            print(f"Warning: No midline value found for year {gl.index(gls)+1}. Taking np.nan")
            midline_x_gl = np.nan
        elif len(midline_x_gl) > 1:
            midline_x_gl = np.min(midline_x_gl)
        else:
            midline_x_gl = midline_x_gl[0]
        midline_x_gls.append(midline_x_gl)

    return midline_x_gls if len(midline_x_gls) > 1 else midline_x_gls[0]

def get_adapted(fpath, iterations="best", years=None):
    simulation_id = fpath.split("-id_")[-1].split(".h5")[0]
    num_subintervals = int(simulation_id.split("_")[2])
    end_time = 200.0
    years_per_subinterval = end_time / num_subintervals
    get_subinterval = lambda year: min(
        int(max(-(year//-years_per_subinterval) - 1, 0)), num_subintervals - 1
    )

    if isinstance(iterations, int | float | np.int64 | np.float64):
        iterations_list = [iterations]
    elif iterations == "best" or iterations == "all":
        iterations_list = get_num_iterations(fpath)
        print(f"Iterations list: {iterations_list}.")
        if iterations == "best":
            min_mass = 1e10
            with CheckpointFile(fpath, "r") as afile:
                _mid_subinterval = get_subinterval(100.0)
                for iter in iterations_list:
                    mesh = afile.load_mesh(f"mesh_iter_{iter}_int_{_mid_subinterval}")
                    h = afile.load_function(
                        mesh, f"thickness_iter_{iter}_int_{_mid_subinterval}_yr_100.0"
                    )
                    _mid_mass = get_mass(h)
                    if _mid_mass < min_mass:
                        min_mass = _mid_mass
                        iterations = [iter]
            print(f"Best iteration for {simulation_id}: {iterations[0]}.")
    else:
        iterations_list = iterations

    if years is not None:
        if isinstance(years, (int, float)):
            years = [years]
    else:
        years = np.arange(0.0, 201.0, 1.0)
        # years = np.arange(1.0, 201.0, 1.0)

    _subintervals = [get_subinterval(year) for year in years]

    hs = []
    us = []
    with CheckpointFile(fpath, "r") as afile:
        for iter in iterations_list:
            _hs, _us = [], []
            _current_subinterval = None
            for ival, year in zip(_subintervals, years):
                if ival != _current_subinterval:
                    mesh = afile.load_mesh(f"mesh_iter_{iter}_int_{ival}")
                    _current_subinterval = ival
                if year == 0.:
                    continue
                    # h = project(h0, FunctionSpace(mesh, "CG", 1), lumped=True)
                    # u = project(u0, VectorFunctionSpace(mesh, "CG", 1), lumped=True)
                else:
                    name_suffix = f"iter_{iter}_int_{ival}_yr_{year}"
                    h = afile.load_function(mesh, f"thickness_{name_suffix}")
                    u = afile.load_function(mesh, f"velocity_{name_suffix}")
                _hs.append(h)
                _us.append(u)

            hs.append(_hs)
            us.append(_us)

    return (hs, us) if len(hs) > 1 else (hs[0], us[0])

def get_metric_density_quotients(metric):
    evalues = []
    evectors = []
    for hess in metric.dat.data:
        eigvals, eigvecs = eig(hess)
        evalues.append(eigvals)
        evectors.append(eigvecs)
    evalues = np.array(evalues)
    evectors = np.array(evectors)

    quotients_x = Function(FunctionSpace(metric.function_space().mesh(), "CG", 1))
    density = quotients_x.copy(deepcopy=True)
    quotients_x.dat.data[:] = np.sqrt(evalues[:, 1] / evalues[:, 0])
    density.dat.data[:] = np.sqrt(evalues[:, 0] * evalues[:, 1])

    return density, quotients_x