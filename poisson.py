import time

from firedrake import *
from animate.metric import RiemannianMetric
from animate.adapt import adapt
from animate.quality import QualityMeasure

import numpy as np
from scipy.stats import linregress

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from firedrake.pyplot import tricontour

from figures import get_label_pos, plt_rcparams, cmap_vik, ax_text

plt.rcParams.update(plt_rcparams)

def solve_poisson(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    u_exact = Function(V)

    eps = 0.01
    f.interpolate(
    - sin(pi * y) * ((2 / eps) * exp(-x / eps) - (1 / eps**2) * (x - 1) * exp(-x / eps)) 
    + pi**2 * sin(pi * y) * ((x - 1) - (x - 1) * exp(-x / eps)))
    u_exact.interpolate((1-exp(-x/eps))*(x-1)*sin(pi*y))

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = f*v*dx
    
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

    u_numerical = Function(V)

    t0 = time.time()
    solve(a == L, u_numerical, bcs=bcs)
    dt = time.time() - t0

    return u_numerical, u_exact, dt

def adapt_poisson(target_complexity, a_max=1e5, h_grad=1.3, n=64, maxiter=5):

    mesh = UnitSquareMesh(n, n)
    u_numerical, *_ = solve_poisson(mesh)

    for i in range(maxiter):
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters(
            {
                "dm_plex_metric_target_complexity": target_complexity,
                "dm_plex_metric_a_max": a_max,
                # "dm_plex_metric_verbosity": 10,
                "dm_plex_metric_gradation_factor": h_grad,
            }
        )
        metric.compute_hessian(u_numerical)
        metric.normalise()
        mesh = adapt(mesh, metric)

        u_numerical, u_exact, dt = solve_poisson(mesh)
        error = errornorm(u_numerical, u_exact)
        print(f"iteration {i}, error = {error:.2e}, dt = {dt:.3f}s")

    num_vertices = mesh.num_vertices()
    print(f"N_v = {num_vertices}, error = {error:.2e}")

    return u_numerical, u_exact, error, metric

def plot_solution(u_n):
    fig, ax = plt.subplots(figsize=(3.27, 3.27))
    # tricontour(u_n, levels=10, axes=ax, colors="k", linewidths=0.8)
    levels = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]
    cs = tricontour(u_n, levels=levels, axes=ax, colors="k", linewidths=0.8)
    print(cs)
    print(cs.levels)
    ax.clabel(cs, colors="k", inline=True, fontsize=6)  # Doesn't work

    # im = tricontourf(u_n, levels=10, axes=ax, cmap=cmap_vik)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # fig.colorbar(im, shrink=0.8)
    fig.savefig("figpoisson1.pdf", bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.01)

def plot_meshes(amesh1, amesh2):
    fig, axes = plt.subplots(2, 2, figsize=(3.27, 4.5), gridspec_kw={"width_ratios": [1, 0.37]})
    # fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    interior_kw = {"linewidth": 0.2}
    boundary_kw = {"color": "k", "linewidth": 0}

    triplot(amesh1, axes=axes[0, 0], interior_kw=interior_kw, boundary_kw=boundary_kw)
    triplot(amesh1, axes=axes[0, 1], interior_kw=interior_kw, boundary_kw=boundary_kw)
    triplot(amesh2, axes=axes[1, 0], interior_kw=interior_kw, boundary_kw=boundary_kw)
    triplot(amesh2, axes=axes[1, 1], interior_kw=interior_kw, boundary_kw=boundary_kw)

    for ax in axes.flatten():
        ax.set_aspect("equal")
        # ax.set_xticks([0, 1])
        # ax.set_yticks([0, 1])
    
    for i, ax in enumerate(axes[:, 0].flatten()):
        # Title
        label = ["(a)", "(b)"][i]
        mesh = [amesh1, amesh2][i]
        qm = QualityMeasure(mesh)
        ar = qm("aspect_ratio")
        print("min ar", ar.dat.data.min(), "max ar", ar.dat.data.max(), "avg ar", ar.dat.data.mean())
        # print("50 largest aspect ratios", sorted(ar.dat.data, reverse=True)[:50])
        
        num_vertices = mesh.num_vertices()
        avg_ar = ar.dat.data.mean()
        ax.set_title(f"{label} $N_v = {num_vertices}$, avg. aspect ratio = ${avg_ar:.2f}$")
        # Draw a box around the zoomed-in region
        ax.plot([0, 0.15, 0.15, 0, 0], [0.3, 0.3, 0.7, 0.7, 0.3], "C0", lw=1.5, zorder=10)
        ax.set_xticks([0, 0.15, 1])
        ax.set_xticklabels(["$0.0$", "$0.15$", "$1.0$"])
        ax.set_yticks([0, 0.3, 0.7, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Label subplots
        # bbox_kw = {"boxstyle": "square,pad=0.1", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"}
        # ax.text(0.02, 0.93, ["(a)", "(b)"][i], bbox=bbox_kw)
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_yticklabels([])

    for ax in axes[:, 1].flatten():
        ax.set_xlim(0, 0.15)
        ax.set_ylim(0.3, 0.7)
        ax.set_xticks([0, 0.15])
        ax.set_yticks([0.3, 0.7])
        # ax.set_xticklabels(["$0$", "$0.15$"])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # axes[0, 1].set_xticklabels([])
    # axes[0, 1].set_yticklabels([])

    # Add ConnectionPatch to the box on the left to zoomed-in region on the right
    for i in range(2):
        con1 = ConnectionPatch(
            xyA=(0., 0.3), xyB=(0.15, 0.3), coordsA="data", coordsB="data",
            axesA=axes[i, 1], axesB=axes[i, 0], color="C0", lw=1.5)
        con2 = ConnectionPatch(
            xyA=(0., 0.7), xyB=(0.15, 0.7), coordsA="data", coordsB="data",
            axesA=axes[i, 1], axesB=axes[i, 0], color="C0", lw=1.5)
        axes[i, 1].add_artist(con1)
        axes[i, 1].add_artist(con2)

    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    # plt.tight_layout()

    fig.savefig("figpoisson2.pdf", bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.01)
    # plt.show()

def plot_adapted_error():
    # Plot error vs number of vertices for different a_max

    fig, ax = plt.subplots(figsize=(3.27, 3.27))

    # Uniform refinement
    ns = [64, 128, 256, 512]
    errors = []
    num_vertices = []
    for n in ns:
        u_n, u_e, _ = solve_poisson(UnitSquareMesh(n, n))
        error = errornorm(u_n, u_e)
        errors.append(error)
        num_vertices.append(u_n.function_space().mesh().num_vertices())
    ax.loglog(num_vertices, errors, ":", marker=".", color="k", label="Uniform")

    # Compute slope for Uniform refinement
    log_nv = np.log(num_vertices)
    log_errors = np.log(errors)
    slope, _, _, _, _ = linregress(log_nv, log_errors)
    print(f"Uniform refinement slope: {slope:.2f}")

    # Adapted refinement
    a_maxs = [1, 2, 4, 8, 16]
    target_complexities = [10, 100, 1000, 10000]
    for i, a_max in enumerate(a_maxs):
        errors = []
        num_vertices = []
        for tc in target_complexities:
            u_n, _, error, _ = adapt_poisson(tc, a_max=a_max, maxiter=2)
            errors.append(error)
            num_vertices.append(u_n.function_space().mesh().num_vertices())
        marker = ["+", "s", "x", "o", "v"][i]
        ax.loglog(num_vertices, errors, marker=marker, fillstyle="none", label="$a_\\mathrm{max}=\\,$"+f"${a_max}$")

        # Compute slope for adapted refinement
        log_nv = np.log(num_vertices)
        log_errors = np.log(errors)
        slope, _, _, _, _ = linregress(log_nv, log_errors)
        print(f"a_max = {a_max}: slope = {slope:.2f}")

    ax.legend()
    ax.grid()
    ax.set_xlabel("$N_v$")
    ax.set_ylabel(r"$\Vert u_h - u \Vert_{L^2}$")
    # ax.set_xlim(3e2, 4e5)
    # ax.set_ylim(1e-7, 2e-4)
    fig.savefig("figpoisson3.pdf", bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.01)




def main():
    # Uniform resolution solution
    # n = 64
    # mesh = UnitSquareMesh(n, n)
    # _, u_e = solve_poisson(mesh)
    # plot_solution(u_e)
    # quit()

    # Adapted solution
    target_complexity = 100
    u_n1, *_ = adapt_poisson(target_complexity, a_max=1, maxiter=2)
    u_n2, *_ = adapt_poisson(target_complexity, a_max=16, maxiter=2)
    mesh1 = u_n1.function_space().mesh()
    mesh2 = u_n2.function_space().mesh()
    plot_meshes(mesh1, mesh2)

    plot_adapted_error()

    # Gradation
    # target_complexity = 200
    # u_n1, *_ = adapt_poisson(target_complexity, h_grad=1.05, a_max=16, maxiter=2)
    # u_n2, *_ = adapt_poisson(target_complexity, h_grad=2.0, a_max=16, maxiter=2)
    # mesh1 = u_n1.function_space().mesh()
    # mesh2 = u_n2.function_space().mesh()
    # plot_meshes(mesh1, mesh2)


if __name__ == "__main__":
    main()
