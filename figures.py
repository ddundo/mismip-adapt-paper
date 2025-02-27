import argparse
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from animate.adapt import adapt
from firedrake import dot, project, sqrt, norm, TensorFunctionSpace, Function, VectorFunctionSpace, FunctionSpace, RectangleMesh, CheckpointFile
from firedrake.pyplot import tripcolor, tricontourf, triplot
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from numpy.linalg import eig

import utility_functions as uf

matplotlib.use('pgf')


def get_label_pos(ax, fig, top=False):
    # Get the aspect ratio of the axis
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    aspect = height / width

    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Check for logarithmic scales
    x_log = ax.get_xscale() == 'log'
    y_log = ax.get_yscale() == 'log'

    x_factor = 0.05
    y_factor = 0.8 if top else 0.1

    if x_log:
        x_min, x_max = np.log10(x_min), np.log10(x_max)
        x_label_pos = 10 ** (x_min + x_factor * (x_max - x_min))
    else:
        x_label_pos = x_min + x_factor * (x_max - x_min) * aspect

    if y_log:
        y_min, y_max = np.log10(y_min), np.log10(y_max)
        y_label_pos = 10 ** (y_min + y_factor * (y_max - y_min) / aspect)
    else:
        y_label_pos = y_min + y_factor * (y_max - y_min)

    return x_label_pos, y_label_pos

def ax_text(ax, fig, text, d=0.07, dy=None, top=False, colour="k"):
    # Get the aspect ratio of the axis
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    aspect = height / width

    x = d
    x *= aspect
    dy = d if dy is None else dy
    y = 1-dy if top else dy

    ax.text(x, y, text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='left',
            color=colour)

def label_axes(fig, axes, top=True, cs=None):
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    if cs is None:
        cs = ["k"] * len(axes)
    for i, ax in enumerate(axes):
        x_label_pos, y_label_pos = get_label_pos(ax, fig, top)
        ax.text(x_label_pos, y_label_pos, panel_labels[i], color=cs[i])

def ice1_base_time_plot(ax):
    # ax.axvline(100, color="k", linestyle=":", alpha=0.4)
    ax.set(
        # xlabel="Years",
        xlim=(0, 200),
    )
    ax.grid()

def get_ref(lv):
    dir_path = f"/data3/glac_adapt/data/reference_outputs/ref_{lv}/analysis"
    vafs = np.load(os.path.join(dir_path, "vafs.npy"))[0]
    gls = np.load(os.path.join(dir_path, "gls.npy"))[0]
    xgls = np.load(os.path.join(dir_path, "midline_x_gls.npy"))[0]
    return vafs, gls, xgls

def get_ref_scatter_kwargs(lv):
    return {
    "s": 10,
    "alpha": 0.8,
    "label": f"{4/2**lv} km",
    "facecolors": "none",
    "edgecolors": f"C{lv}",
}

plt_rcparams = {
    "pgf.texsystem": "pdflatex",

    "font.size": 8,
    "font.family": "sans-serif",
    "font.serif": "DejaVu Sans",

    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{bm}",

    "axes.titlesize": 8,

    "legend.columnspacing": 0.8,
    "legend.handletextpad": 0.3,
    # "legend.markerscale": 0.7,
    "legend.borderpad": 0.,
    "legend.frameon": False,

    # space between colorbar and label
    "axes.labelpad": 2,
    "axes.titlepad": 3.5,

    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "inout",
    "ytick.direction": "inout",
    # xtick label distance from tick
    "xtick.major.pad": 1.,
    "ytick.major.pad": 1.,

    # colorbar ticks
    "ytick.minor.width": 0.3,
    "ytick.minor.size": 1.2,
    "axes.linewidth": 0.5,
    # "legend.handlelength": 5.0,

    "lines.markersize": 4,
    "lines.linewidth": 0.8,

    # grid opacity 0.3
    "grid.alpha": 0.3,
}
plt.rcParams.update(plt_rcparams)

# read Scientific Colour Maps (https://github.com/callumrollo/cmcrameri)
cmap_lipari = LinearSegmentedColormap.from_list('lipari', np.loadtxt("cmcrameri/lipari.txt"))
cmap_oleron = LinearSegmentedColormap.from_list('oleron', np.loadtxt("cmcrameri/oleron.txt"))
cmap_devon = LinearSegmentedColormap.from_list('devon', np.loadtxt("cmcrameri/devon.txt"))
cmap_devon_reversed = LinearSegmentedColormap.from_list('devon_reversed', np.loadtxt("cmcrameri/devon.txt")[::-1])
cmap_devonS = LinearSegmentedColormap.from_list('devonS', np.loadtxt("cmcrameri/devonS.txt"))
cmap_devonS = LinearSegmentedColormap.from_list('devonS', np.loadtxt("cmcrameri/devonS.txt"))
cmap_vik = LinearSegmentedColormap.from_list('vik', np.loadtxt("cmcrameri/vik.txt"))

def plot_metric_components(args):
    initial_steady_state_path = os.path.join(args.output_dir, "steady_state", "outputs-Ice1-id_steady_state.h5")
    with CheckpointFile(initial_steady_state_path, "r") as afile:
        old_mesh = afile.load_mesh("firedrake_default")
        u = afile.load_function(old_mesh, "velocity_steady")
        h = afile.load_function(old_mesh, "thickness_steady")

    mp = {
        "dm_plex_metric": {
            "target_complexity": 1600,
            "h_min": 1.0,
            "h_max": 50e3,
            "p": 2.0,
            "a_max": 1e30,
        }
    }
    h_metric = uf.h_metric(h, u, mp)
    h_metric.normalise()
    u_metric = uf.u_metric(h, u, mp)
    u_metric.normalise()
    tau_metric = uf.tau_metric(h, u, mp)
    tau_metric.normalise()

    # zoom in bounds ([x_min, x_max, y_min, y_max])
    bounds = [[135e3, 165e3, 12e3, 20e3], [470e3, 485e3, 14e3, 18e3], [442e3, 458e3, 30e3, 34.3e3]]

    interior_kw = dict(linewidth=0.05, alpha=0.9, rasterized=True)
    interior_zoom_kw = dict(linewidth=0.1, rasterized=True)
    boundary_kw = dict(linewidth=0., color="k")
    tripcolor_kw = dict(cmap=cmap_vik, shading="flat", rasterized=True)

    gridspec_kw = {'height_ratios': [1, 0.5, 1, 0.5, 1, 0.65]}
    fig, axes = plt.subplots(6, 3, figsize=(4.72, 3.5), gridspec_kw=gridspec_kw)
    for i, metric in enumerate([h_metric, u_metric, tau_metric]):
        density, quotients = uf.get_metric_density_quotients(metric)

        im0 = tripcolor(density, axes=axes[0, i], norm=LogNorm(vmin=1e-9, vmax=1e-5), **tripcolor_kw)
        im1 = tripcolor(quotients, axes=axes[2, i], norm=LogNorm(vmin=1e-2, vmax=1e2), **tripcolor_kw)

        # Add only one colorbar per row
        if i == 0:
            cbar0 = fig.colorbar(im0, ax=axes[1, :], orientation="horizontal", aspect=50, location="bottom")
            cbar0.set_label(r"Metric density $\rho$")
            cbar1 = fig.colorbar(im1, ax=axes[3, :], orientation="horizontal", pad=0.15, aspect=50, shrink=1.0, location="bottom")
            cbar1.set_label(r"Anisotropy quotient $r_1$")
            for cidx, cbar in enumerate([cbar0, cbar1]):
                cbar.ax.xaxis.set_label_position('top')
                for c in cbar.ax.collections:
                    c.set_rasterized(True)
                # reposition cbar slightly up
                pos = cbar.ax.get_position()
                pos.y0 += 0.025 if cidx == 0 else 0.033
                pos.y1 += 0.025 if cidx == 0 else 0.033
                cbar.ax.set_position(pos)

        adapted_mesh = adapt(old_mesh, metric)
        triplot(adapted_mesh, axes=axes[4, i], interior_kw=interior_kw, boundary_kw=boundary_kw)
        triplot(adapted_mesh, axes=axes[5, i], interior_kw=interior_zoom_kw, boundary_kw=boundary_kw)

        # draw zoom in boxes
        x_min, x_max, y_min, y_max = bounds[i]
        axes[4, i].plot(
            [x_min, x_max, x_max, x_min, x_min], 
            [y_min, y_min, y_max, y_max, y_min], 
            "--",
            color="C0", 
            linewidth=0.5,
            path_effects=[pe.Stroke(linewidth=1.0, foreground='black'), pe.Normal()]
        )

    # Add titles for each column
    for i, ax in enumerate(axes[0, :]):
        ax.set_title("Sensor field: " + [r"$H$", r"$\mathbf{u}$", r"$\tau_b$"][i])

    for i in range(6):
        for j, ax in enumerate(axes[i, :]):
            if i in [0, 2, 4, 5]:
                ax.set_xlim(0, 640e3)
                ax.set_ylim(0, 40e3)
                ax.set_xticks([j*100e3 for j in range(7)] + [640e3])
                ax.set_yticks([0, 40e3])
            if i == 5:
                x_min, x_max, y_min, y_max = bounds[j]
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect("equal")
            ax.set_xticklabels([])  # Clear x-tick labels
            ax.set_yticklabels([])  # Clear y-tick labels
        if i in [1, 3]:
            for ax in axes[i, :]:
                ax.axis("off")
    fontdict = {"fontsize": 6}
    axes[0, 0].set_xticklabels(["$0$", "", "", "", "", "", "", "$640$"], fontdict=fontdict)
    axes[0, 0].set_yticklabels(["-$40$", "$0$"], fontdict=fontdict)
    axes[0, 0].set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-6, fontdict=fontdict)
    axes[0, 0].set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-10, fontdict=fontdict)

    # Connect the bounding boxes in the second to last row with the last row
    for i in range(3):
        x_min, x_max, y_min, y_max = bounds[i]
        con = ConnectionPatch(xyA=(x_min, y_min), xyB=(x_min, y_max), coordsA="data", coordsB="data",
                              axesA=axes[4, i], axesB=axes[5, i],
                              linestyle="--",
                              arrowstyle="-", lw=0.5, color="C0",
                              path_effects=[pe.Stroke(linewidth=1.0, foreground='black'), pe.Normal()])
        axes[4, i].add_artist(con)
        con = ConnectionPatch(xyA=(x_max, y_min), xyB=(x_max, y_max), coordsA="data", coordsB="data",
                                axesA=axes[4, i], axesB=axes[5, i], linestyle="--",
                                arrowstyle="-", lw=0.5, color="C0",
                                path_effects=[pe.Stroke(linewidth=1.0, foreground='black'), pe.Normal()])
        axes[4, i].add_artist(con)
        

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0., wspace=0.02)

    return fig

def plot_initial_steady_state(args):
    initial_steady_state_path = os.path.join(args.output_dir, "steady_state", "outputs-Ice1-id_steady_state.h5")
    with CheckpointFile(initial_steady_state_path, "r") as afile:
        old_mesh = afile.load_mesh("firedrake_default")
        u = afile.load_function(old_mesh, "velocity_steady")
        h = afile.load_function(old_mesh, "thickness_steady")
    z_b = Function(FunctionSpace(old_mesh, "CG", 1)).interpolate(uf.mismip_bed_topography(old_mesh))
    s = Function(h.function_space()).interpolate(uf.surface_expression(h, z_b))
    friction = uf.friction_law(thickness=h, velocity=u, surface=s, friction=1e-2)
    tau_b = Function(u.function_space()).interpolate(-diff(friction, u))
    tau_b.dat.data[:] *= 1e3
    gl = uf.get_gl(uf.get_haf(h))

    fig, axes = plt.subplots(3, 1, figsize=(3.27, 3.0), sharex=True)
    tricontourf_kw = dict(cmap=cmap_vik, levels=9)
    cbar_labels = [r"$H$ ($\mathrm{m}$)", r"$\mathbf{u}$ ($\mathrm{ma^{-1}}$)", r"$\tau_b$ ($\mathrm{Pa}$)"]
    for i, (ax, f) in enumerate(zip(axes, [h, u, tau_b])):
        cf = tricontourf(f, axes=ax, **tricontourf_kw)
        cf.set_rasterized(True)

        z_b_data = z_b.dat.data
        x_coords, y_coords = old_mesh.coordinates.dat.data.T
        cs = ax.tricontour(x_coords, y_coords, z_b_data, 
                           levels=[-605, -405, -205, 5, 205],
                           colors="white", linewidths=0.1, alpha=0.7)
        ax.clabel(cs, colors="white", inline=True, fontsize=6)
        ax.plot(gl[:, 0], gl[:, 1], 'y-', linewidth=1, label="Grounding line",
                path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

        cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.15, aspect=30, shrink=0.7, location="top")
        for c in cbar.ax.collections:
            c.set_rasterized(True)
        cbar.set_label(cbar_labels[i])
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('top')

        if i == 0:
            # Hide every second tick label for the first colorbar
            tick_labels = [label.get_text() if j % 2 == 0 else '' for j, label in enumerate(cbar.ax.get_xticklabels())]
            cbar.ax.set_xticklabels(tick_labels)

        ax.set_xlim(0, 640e3)
        ax.set_ylim(0, 40e3)
        ax.set_xticks([j*100e3 for j in range(7)] + [640e3])
        ax.set_xticklabels([] if ax != axes[-1] else ["$0$"] + [""]*6 + ["$640$"])
        ax.set_yticks([0, 40e3])
        ax.set_yticklabels([] if ax != axes[-1] else ["-$40$", "$0$"])

    axes[-1].set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-8)
    axes[-1].set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-12)

    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax_text(ax, fig, label, dy=0.3, top=True, colour="white")

    # Custom legend entry for bed topography z_b
    z_b_line = Line2D([0], [0], color='#b3b3b3', linewidth=0.1, alpha=0.5, label=r"Topography contours ($\mathrm{m}$)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -1.8), ncol=2, frameon=False,
                   handles=[z_b_line, *axes[1].get_legend_handles_labels()[0]])

    fig.subplots_adjust(hspace=0.15)

    return fig

def plot_uniform_convergence(args):
    Lx = 640e3
    Ly = 40e3
    ny = 40
    msh = RectangleMesh(int(Lx/Ly)*ny, ny, Lx, Ly)
    msh.coordinates.dat.data[:] /= 1e3

    # Reference solutions
    years = np.arange(1., 201., 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.72, 2.7), sharex=False)

    # dictionary mapping level of refinement to mesh resolution in km
    levels_res = {0: 4, 1: 2, 2: 1, 3: 0.5, 4: 0.25, 5: 0.125}

    for lv in levels_res.keys():
        res_m = int(levels_res[lv] * 1000)
        dir_path = os.path.join(args.output_dir, f"uniform_{res_m}", "analysis")
        vafs = np.load(os.path.join(dir_path, "vafs.npy"))[0] / 1e3  # 1000 km^3
        midline_x_gls = np.load(os.path.join(dir_path, "midline_x_gls.npy"))[0] / 1e3

        plot_kwargs = dict(
            c=f"C{lv}",
            label=fr"${levels_res[lv]}\ \mathrm{{km}}$",
            linestyle="--" if lv==5 else "-",
        )

        axes[0, 0].plot(years, vafs, **plot_kwargs,)
        axes[1, 0].plot(years, midline_x_gls, **plot_kwargs)

        for i in range(2):
            yr = (i+1)*100-1
            gl = np.load(os.path.join(dir_path, "gls.npy"))[0][yr] / 1e3
            # tidy up the gl
            gl = np.round(gl / levels_res[lv]) * levels_res[lv]
            gl = uf.tidy_gl(gl)
            # subtract 40km from all y positions of gl
            gl[:, 1] -= 40

            axes[i, 1].plot(gl[:, 0], gl[:, 1], **plot_kwargs)

            axes[i, 1].set_ylim(-40, 0)
            axes[i, 1].set_xlim(380, 520)
            axes[i, 1].grid(True)

            # axes[i, 1].yaxis.set_major_locator(MaxNLocator(nbins=2))

    ice1_base_time_plot(axes[0, 0])
    ice1_base_time_plot(axes[1, 0])

    axes[0, 0].set_ylim(15, 17)  # vaf
    axes[1, 0].set_ylim(380, 460)

    # axes[0, 0].set_title("Ice mass (Gt)")
    axes[0, 0].set_ylabel(r"$V_f\ (10^3\ \mathrm{km}^3$)")

    # axes[1, 0].set_title(r"Midchannel g.l. position ($y\!=\!0\ \mathrm{km}$)")
    axes[1, 0].set_ylabel(r"$x_{gl}(y=0)\ (\mathrm{km})$")

    # axes[0].set_xlabel("Years")
    axes[1, 0].set_xlabel(r"Time ($\mathrm{a}$)")
    axes[1, 1].set_xlabel(r"$x\ (\mathrm{km})$")

    for ax in axes[:, 1]:
        ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-13)
        ax.set_yticks([-10*(4-jj) for jj in range(5)])
        ax.set_yticklabels(["$-40$", "", "", "", "$0$"])

    titles = [
        "Volume above flotation",
        "Midchannel grounding line position",
        r"Grounding line at $t\!=\!100\ \mathrm{a}$",
        r"Grounding line at $t\!=\!200\ \mathrm{a}$"
    ]
    for title, ax in zip(titles, axes.flatten(order="F")):
        ax.set_title(title, pad=1)

    panel_labels = [["(a)", "(c)"], ["(b)", "(d)"]]
    panel_labels_pos = [[(5, 55.5), (385, 39)], [(5, 385), (385, 5)]]
    for i in range(2):
        for j in range(2):
            x_label_pos, y_label_pos = get_label_pos(axes[i, j], fig)
            axes[i, j].text(x_label_pos, y_label_pos, panel_labels[i][j])
            # axes[i, j].text(*panel_labels_pos[i][j], panel_labels[i][j])

    for ax in axes.flatten():
        # set distance from tick label to tick to 1
        ax.tick_params(pad=1)

    # turn off x ticks for top plots
    for ax in axes[0]:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    # add a legend below the plot
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.03, -1.5),
        ncol=5,
        title="Mesh resolution",
        # distance between labels
        columnspacing=1.2,
        # frameon=False,
    )

    fig.subplots_adjust(wspace=0.15, hspace=0.2)

    return fig

def plot_metric_intersection(args):
    initial_steady_state_path = os.path.join(args.output_dir, "steady_state", "outputs-Ice1-id_steady_state.h5")
    with CheckpointFile(initial_steady_state_path, "r") as afile:
        old_mesh = afile.load_mesh(f"firedrake_default")
        u = afile.load_function(old_mesh, "velocity_steady")
        h = afile.load_function(old_mesh, "thickness_steady")
    mp = {
        "dm_plex_metric": {
            "target_complexity": 1600,
            "h_min": 1.0,
            "h_max": 50e3,
            "p": 2.0,
            "a_max": 1e30,
        }
    }
    uh_metric = uf.uh_metric(h, u, mp)
    uh_metric.normalise()

    interior_kw = dict(linewidth=0.05, alpha=0.9, rasterized=True)
    boundary_kw = dict(linewidth=0., color="k")
    tripcolor_kw = dict(cmap=cmap_vik, shading="flat", rasterized=True)

    fig, axes = plt.subplots(3, 1, figsize=(3.27, 3.0), sharex=True)
    density, quotients = uf.get_metric_density_quotients(uh_metric)

    im0 = tripcolor(density, axes=axes[0], norm=LogNorm(vmin=1e-9, vmax=1e-5), **tripcolor_kw)
    im1 = tripcolor(quotients, axes=axes[1], norm=LogNorm(vmin=1e-2, vmax=1e2), **tripcolor_kw)

    cbar0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.15, aspect=30, shrink=0.7, location="top")
    cbar0.set_label(r"Metric density $\rho$")
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.15, aspect=30, shrink=0.7, location="top")
    cbar1.set_label(r"Anisotropy quotient $r_1$")
    for cbar in [cbar0, cbar1]:
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        for c in cbar.ax.collections:
            c.set_rasterized(True)
        # reposition cbar slightly up
        pos = cbar.ax.get_position()
        pos.y0 += 0.02
        pos.y1 += 0.02
        cbar.ax.set_position(pos)

    adapted_mesh = adapt(old_mesh, uh_metric)
    triplot(adapted_mesh, axes=axes[2], interior_kw=interior_kw, boundary_kw=boundary_kw)

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 640e3)
        ax.set_ylim(0, 40e3)
        ax.set_xticks([j*100e3 for j in range(7)] + [640e3])
        ax.set_xticklabels([] if ax != axes[-1] else ["$0$"] + [""]*6 + ["$640$"])
        ax.set_yticks([0, 40e3])
        ax.set_yticklabels([] if ax != axes[-1] else ["-$40$", "$0$"])
        ax_text(ax, fig, f"({chr(97+i)})", dy=0.31, top=True, colour="black")

    # move last axis slightly up
    pos = axes[-1].get_position()
    pos.y0 += 0.08
    pos.y1 += 0.01
    axes[-1].set_position(pos)

    axes[-1].set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-8)
    axes[-1].set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-12)

    return fig

def plot_strat_comparison_conv(args):
    # set marker size to 2
    plt.rcParams.update({"lines.markersize": 2})
    colors = cmap_oleron(np.linspace(0, 1, 14))

    ref_dirpath = os.path.join(args.output_dir, "uniform_250", "analysis")
    glob_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_global", "analysis")
    glob_tau_3200 = os.path.join(args.output_dir, "3200_tau_20_global", "analysis")
    hybr_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_hybrid", "analysis")
    hybr_tau_3200 = os.path.join(args.output_dir, "3200_tau_20_hybrid", "analysis")
    fpaths = [glob_tau_1600, glob_tau_3200, hybr_tau_1600, hybr_tau_3200]
    labels = [r"Global, $\mathcal{C}=1600$", r"Global, $\mathcal{C}=3200$",
              r"Hybrid, $\mathcal{C}=1600$", r"Hybrid, $\mathcal{C}=3200$"]

    # reference volume above flotation and midchannel grounding line position
    ref_vafs = np.load(os.path.join(ref_dirpath, "vafs.npy"))[0, 1:]
    ref_xgls = np.load(os.path.join(ref_dirpath, "midline_x_gls.npy"))[0, 1:]

    # first iteration of the global algorithm is the same as uniform 4000m simulation
    ref_4000m_dirpath = os.path.join(args.output_dir, "uniform_4000", "analysis")
    ref_lv0_vafs = np.load(os.path.join(ref_4000m_dirpath, "vafs.npy"))[0, 1:]
    ref_lv0_xgls = np.load(os.path.join(ref_4000m_dirpath, "midline_x_gls.npy"))[0, 1:]
    diff_vafs_lv0 = np.linalg.norm(ref_vafs - ref_lv0_vafs, np.inf)
    diff_xgls_lv0 = np.linalg.norm(ref_xgls - ref_lv0_xgls, np.inf)

    fig, axes = plt.subplots(2, 1, figsize=(3.27, 2.5), sharex=True)

    for i in range(len(fpaths)):

        vafs = np.load(f"{fpaths[i]}vafs.npy")
        xgls = np.load(f"{fpaths[i]}midline_x_gls.npy")

        for j in range(len(xgls)):
            for k in range(1, len(xgls[j])-1):
                if np.isnan(xgls[j, k]):
                    xgls[j, k] = (xgls[j, k-1] + xgls[j, k+1]) / 2

        diff_vafs, diff_xgls = [], []
        if i < 2:
            diff_vafs.append(diff_vafs_lv0)
            diff_xgls.append(diff_xgls_lv0)

        for j in range(len(vafs)):
            diff_vafs.append(np.linalg.norm(vafs[j] - ref_vafs, np.inf))
            diff_xgls.append(np.linalg.norm(xgls[j] - ref_xgls, np.inf))

        # convert xgls to km
        diff_xgls = np.array(diff_xgls) / 1e3

        num_iters = list(range(1, len(diff_vafs)+1))

        for ax, n in zip(axes, [diff_vafs, diff_xgls]):
            color = colors[10] if i < 2 else colors[1]
            linestyle = "-" if i % 2 == 0 else "--"
            fillstyle = "full"
            ax.semilogy(num_iters, n, 'o-', label=labels[i], color=color, linestyle=linestyle, fillstyle=fillstyle)

    dofs = np.load(f"{fpaths[-2]}dofs.npy")
    dofs = dofs[:-1]
    for i, dof in enumerate(dofs, start=1):
        if i in (1, 2, 4, 6):
            axes[-1].plot(dof, label=f"{i}", color=colors[6-i])
    dofs_glob = np.load(f"{fpaths[0]}dofs.npy")[-1]
    axes[-1].plot(dofs_glob, label="10", color=colors[10],)

    xlims = [(1, 10), (1, 10)]
    ylims = [(1e1, 2e3), (1e0, 1e2)]
    xlabels = ["", "Iteration $k$"]
    ylabels = [r"$||\Delta V_f||_\infty \ (\mathrm{km}^3)$", 
            r"$||\Delta x_{gl}||_\infty \ (\mathrm{km})$",]
    ncols = [1, 1]

    axes[0].set_title("Convergence of global and hybrid algorithms")

    for i, ax in enumerate(axes):
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        if i != 2:
            ax.set_xticks(np.arange(1, 11))
        if i == 0:
            ax.legend(ncol=ncols[i])
        ax.grid()
        x_label_pos, y_label_pos = get_label_pos(ax, fig, top=False)
        if i == 0:
            y_label_pos -= 31
        elif i == 1:
            y_label_pos -= 2.4
        ax.text(x_label_pos, y_label_pos, ["(a)", "(b)"][i])

    fig.align_labels()
    fig.subplots_adjust(hspace=0.1)

    return fig

def plot_strat_comparison_meshes(args):
    glob_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_global", "outputs-Ice1-id_1600_tau_20_global.h5")
    hybr_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_hybrid", "outputs-Ice1-id_1600_tau_20_hybrid.h5")
    fpaths = [glob_tau_1600, hybr_tau_1600]

    fig_labels = ["(a)", "(b)", "(c)", "(d)"]

    gridspec_kw = dict(hspace=0.0, wspace=0.01, width_ratios=[1, 0.7])
    fig, axes = plt.subplots(4, 2, figsize=(4.72, 3.5), sharex='col', sharey='row', gridspec_kw=gridspec_kw)
    interior_kw = {"linewidth": 0.05, "alpha": 0.9, "rasterized": True}

    for i in range(len(fpaths)):
        glob = i == 0
        niter = 10 if i == 0 else 6
        with CheckpointFile(fpaths[i], "r") as afile:
            mesh_9 = afile.load_mesh(f"mesh_iter_{niter}_int_9")
            mesh_19 = afile.load_mesh(f"mesh_iter_{niter}_int_19")

        for j, mesh in enumerate([mesh_9, mesh_19]):
            ax = axes[i + j * 2, 0]
            triplot(mesh, axes=ax, interior_kw=interior_kw)

            mesh_interval = 9 if j == 0 else 19
            mesh_label = fr"$\mathcal{{H}}_{{{mesh_interval}}}^{{({niter})}}$"
            alg_label = "Global" if glob else "Hybrid"
            ax_text(ax, fig, f"{fig_labels[i + j * 2]} {alg_label}, {mesh_label}", dy=0.25, top=True)
            ax_text(ax, fig, f"$N_v={mesh.num_vertices()}$", top=False)

            ax.set_yticks([0, 40e3])
            ax.set_yticklabels([] if ax != axes[-1, 0] else ["-$40$", "$0$"])
            ax.set_xlim(0, 640e3)
            ax.set_ylim(0, 40e3)

            # Plot the bounding box around the zoomed-in region
            ax.plot([380e3, 520e3, 520e3, 380e3, 380e3], [0, 0, 40e3, 40e3, 0], '--',
                    color="C0", linewidth=0.5, rasterized=True,
                    path_effects=[pe.Stroke(linewidth=1.0, foreground='black'), pe.Normal()])

            # Plot the zoomed-in region
            ax_zoom = axes[i + j * 2, 1]
            triplot(mesh, axes=ax_zoom, interior_kw=interior_kw)
            ax_zoom.set_xlim(380e3, 520e3)
            ax_zoom.set_ylim(0, 40e3)
            ax_zoom.set_xticks([400e3, 500e3])
            ax_zoom.set_yticks([0, 40e3])
            ax_zoom.set_xticklabels([])
            ax_zoom.set_yticklabels([])

    # Set labels and adjust positions for the last row
    ax = axes[-1, 0]
    ax.set_xticks([0, 100e3, 200e3, 300e3, 400e3, 500e3, 600e3, 640e3])
    ax.set_xticklabels(["$0$"] + [""] * 6 + ["$640$"])
    ax.set_yticks([0, 40e3])
    ax.set_yticklabels(["-$40$", "$0$"])
    ax.set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-8)
    ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-10)
    ax.xaxis.set_tick_params(pad=1)
    ax.yaxis.set_tick_params(pad=1)

    dy = 0.01
    for j in range(2):
        for i, ax in enumerate(axes[:, j]):
            pos = ax.get_position()
            if i == 0:
                pos.y0 += dy
            elif i == 1:
                pos.y0 += dy
                pos.y1 += dy
            elif i == 2:
                pos.y0 -= dy
                pos.y1 -= dy
            elif i == 3:
                pos.y1 -= dy
            ax.set_position(pos) 

    ax_zoom = axes[-1, 1]
    ax_zoom.xaxis.set_tick_params(pad=1)

    return fig

def plot_strat_comparison_dofs(args):
    # set marker size to 2
    plt.rcParams.update({"lines.markersize": 2})
    colors = cmap_oleron(np.linspace(0, 1, 14))

    glob_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_global", "analysis")
    glob_tau_3200 = os.path.join(args.output_dir, "3200_tau_20_global", "analysis")
    hybr_tau_1600 = os.path.join(args.output_dir, "1600_tau_20_hybrid", "analysis")
    hybr_tau_3200 = os.path.join(args.output_dir, "3200_tau_20_hybrid", "analysis")
    fpaths = [glob_tau_1600, glob_tau_3200, hybr_tau_1600, hybr_tau_3200]
    labels = [r"Global, $\mathcal{C}=1600$", r"Global, $\mathcal{C}=3200$",
              r"Hybrid, $\mathcal{C}=1600$", r"Hybrid, $\mathcal{C}=3200$"]

    fig, ax = plt.subplots(figsize=(3.27, 1.5))

    dofs = np.load(f"{fpaths[-2]}dofs.npy")
    dofs = dofs[:-1]
    for i, dof in enumerate(dofs, start=1):
        if i in (1, 2, 4, 6):
            ax.plot(dof, label=f"{i}", color=colors[6-i])
    dofs_glob = np.load(f"{fpaths[0]}dofs.npy")[-1]
    ax.plot(dofs_glob, label="10", color=colors[10],)

    ax.set_title(r"Temporal distribution of $N_v$ ($\mathcal{C}=1600$)")
    ax.set_xlim(0, 200)
    ax.set_ylim(1750, 4000)
    ax.set_xlabel("Time $(a)$")
    ax.set_ylabel("$N_v$")
    ax.set_yticks(np.arange(1750, 3751, 500))
    ax.legend(ncol=2)
    handles, labels = ax.get_legend_handles_labels()
    lg0 = ax.legend(handles[:4], labels[:4], ncol=2, title="Iteration (hybrid)",)
    lg1 = ax.legend(handles[4:], labels[4:], ncol=1, title="Iteration (global)",)
    for lg in [lg0, lg1]:
        lg._legend_box.sep = 0
    ax.add_artist(lg0)
    lg1.set_bbox_to_anchor((0.63, 0.3))
    ax.grid()

    return fig

def plot_h_field_comparison(args):
    categorical_colors = cmap_devonS(np.linspace(0, 1, 11))

    # ref4_fpath = "/data3/glac_adapt/data/reference_outputs/ref_4/analysis/gls.npy"
    ref4_fpath = os.path.join(args.output_dir, "uniform_250", "analysis", "gls.npy")
    ref4_gl_90 = np.load(ref4_fpath)[0, 89]
    ref4_gl_100 = np.load(ref4_fpath)[0, 99]

    cs = [800*2**i for i in range(5)]

    fig = plt.figure(figsize=(4.72, 4))
    gs = gridspec.GridSpec(11, 2,
                           figure=fig,
                           hspace=0.2,
                           wspace=0.05,
                           width_ratios=[1, 1.5], height_ratios=[1] * 10 + [0.3])

    fields = ["h", "u", "u-int-s", "s", "tau"]
    labels = {"h": "$h$", "tau": r"$\tau_b$", "u": "$u$", "u-int-h": r"$(h,\,u)$", "s": "$s$", "u-int-s": r"$(u,\,s)$"}

    for i, field in enumerate(fields):

        d, n_h, n_umag = [], [], []
        n_vaf, n_xgl = [], []
        for c in cs:
            analysis_path = os.path.join(args.output_dir, f"{int(c)}_{field}_20_hybrid", "analysis")
            norms = np.load(os.path.join(analysis_path, "norms.npy"))
            dofs = np.load(os.path.join(analysis_path, "dofs.npy"))

            norms_avg_time = np.mean(norms, axis=2)
            norms_avg_fields = np.mean(norms_avg_time, axis=0)
            min_iter_idx = np.argmin(norms_avg_time[1])

            if c == 6400:
                simulation_id = f"{c}_{field}_20_hybrid"
                fpath = os.path.join(args.output_dir, f"{simulation_id}", f"outputs-Ice1-id_{simulation_id}.h5")
                with CheckpointFile(fpath, "r") as afile:
                    msh = afile.load_mesh(f"mesh_iter_{min_iter_idx}_int_9")
                ax = plt.subplot(gs[(i*2):(i*2+2), 1])
                triplot(msh, axes=ax, interior_kw={"linewidth": 0.05,
                                                #    "alpha": 0.2,
                                                    "rasterized": True
                                                    })
                ax.plot(ref4_gl_90[:, 0], ref4_gl_90[:, 1], 'y--', linewidth=0.5, rasterized=True, label=r"$t=90\ \mathrm{a}$")
                ax.plot(ref4_gl_100[:, 0], ref4_gl_100[:, 1], 'y-', linewidth=0.5, rasterized=True, label=r"$t=100\ \mathrm{a}$")
                # ax.text(10e3, 2e3, f"{field}, {msh.num_vertices()} DoFs")
                ax.set_xlim(0, 640e3)
                ax.set_ylim(0, 40e3)
                ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
                ax.set_yticks([0, 40e3])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                x_label_pos, y_label_pos = get_label_pos(ax, fig, top=True)
                ax.text(x_label_pos, y_label_pos, ["(c)", "(d)", "(e)", "(f)", "(g)"][i])
                if i == 4:
                    ax.set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-6)
                    ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-10)
                    ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
                    ax.set_xticklabels(["$0$"] + [""]*6 + ["$640$"])
                    ax.set_yticklabels(["$-40$", "$0$"])
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    # increase the distance from x tick labels to the plot
                    ax.xaxis.set_tick_params(pad=5.5)

            d.append(np.mean(dofs, axis=1)[min_iter_idx])
            n_h.append(norms_avg_time[0, min_iter_idx])
            n_umag.append(norms_avg_time[1, min_iter_idx])

        ax00 = plt.subplot(gs[0:5, 0])
        ax10 = plt.subplot(gs[5:10, 0])
        ax00.loglog(d, n_h, 'o-', label=labels[field], color=categorical_colors[i])
        ax10.loglog(d, n_umag, 'o-', label=labels[field], color=categorical_colors[i])

    ref_d, ref_n_h, ref_n_umag = [], [], []
    for res in 4000, 2000, 1000, 500, 250:
        norms = np.load(os.path.join(args.output_dir, f"uniform_{res}", "analysis", "norms.npy"))
        ref_d.append(np.load(fpath.replace("norms", "dofs"))[0, 0])
        norms_avg_time = np.mean(norms, axis=1)
        ref_n_h.append(norms_avg_time[0])
        ref_n_umag.append(norms_avg_time[1])

    ax00.loglog(ref_d, ref_n_h, 'o:', label="Uniform", color='k')
    ax10.loglog(ref_d, ref_n_umag, 'o:', label="Uniform", color='k')

    for i, (_ax, n) in enumerate(zip([ax00, ax10], [n_h, n_umag])):
        _ax.set_xlim(1e3, 1.2e5)
        _ax.grid()
        x_label_pos, y_label_pos = get_label_pos(_ax, fig)
        _ax.text(x_label_pos, y_label_pos, ["(a)", "(b)"][i])
    ax00.set_xticklabels([])
    ax00.set_ylim(1e-3, 2e-1)
    ax10.set_ylim(1e-2, 1e-0)
    ax10.set_xlabel("Average $N_v$")

    ax00.set_ylabel(r"$\tilde{e}_h$")
    ax10.set_ylabel(r"$\tilde{e}_u$")

    # reposition ax00 a bit more to above and ax10 a bit more down
    pos = ax00.get_position()
    pos.y0 += 0.008
    ax00.set_position(pos)
    pos = ax10.get_position()
    pos.y1 -= 0.008
    ax10.set_position(pos)

    # create an axis below the bottom one for the legend
    ax0_legend = plt.subplot(gs[10, 0])
    ax0_legend.axis("off")
    handles = ax00.get_legend_handles_labels()[0]
    labels = ax00.get_legend_handles_labels()[1]

    leg1 = ax0_legend.legend(handles[:3], labels[:3], loc='lower center', ncol=3, frameon=False)
    leg2 = ax0_legend.legend(handles[3:], labels[3:], loc='upper center', ncol=3, frameon=False)
    ax0_legend.add_artist(leg1)

    legend_pos = ax0_legend.get_position()
    legend_pos.y0 -= 0.13
    legend_pos.y1 -= 0.13
    ax0_legend.set_position(legend_pos)

    legend_ax = plt.subplot(gs[10, 1])
    legend_ax.axis('off')
    legend_ax.set_title("Reference g.l. contours", fontsize=7, pad=6)
    handles = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]
    legend_ax.legend(handles=handles[4:], labels=labels[4:], loc='center', ncol=2)
    # move legend_ax below
    legend_pos = legend_ax.get_position()
    legend_pos.y0 -= 0.13
    legend_pos.y1 -= 0.13
    legend_ax.set_position(legend_pos)

    fig.align_labels()

    return fig

def plot_hessian_meshes(args):
    ref4_fpath = os.path.join(args.output_dir, "uniform_250", "analysis", "gls.npy")
    ref4_gl_90 = np.load(ref4_fpath)[0, 89]
    ref4_gl_100 = np.load(ref4_fpath)[0, 99]

    c = 6400
    fields = ["h", "u", "u-int-h", "tau"]
    labels = {"h": "$H$", "tau": r"$\tau_b$", "u": r"$\mathbf{u}$", "u-int-h": r"$(H,\,\mathbf{u})$"}
    subplot_labels = ["(a)", "(b)", "(c)", "(d)",]

    fig, axes = plt.subplots(len(fields), 2, figsize=(4.72, 3), 
                             sharex='col', sharey='row', 
                             gridspec_kw={"width_ratios": [1, 0.7], "wspace": 0.01, "hspace": 0.05})

    for i, field in enumerate(fields):
        simdir_path = os.path.join(args.output_dir, f"{c}_{field}_20_hybrid/")
        fpath = os.path.join(simdir_path, f"outputs-Ice1-id_{c}_{field}_20.h5")
        with CheckpointFile(fpath, "r") as afile:
            msh = afile.load_mesh(f"mesh_iter_4_int_9")

        interior_kw = {"linewidth": 0.05, "alpha": 0.9, "rasterized": True}

        # Plot the full domain
        ax = axes[i, 0]
        triplot(msh, axes=ax, interior_kw=interior_kw)
        ax.plot(ref4_gl_90[:, 0], ref4_gl_90[:, 1], 'y--', linewidth=0.5, rasterized=True, label="$t=90\ \mathrm{a}$")
        ax.plot(ref4_gl_100[:, 0], ref4_gl_100[:, 1], 'y-', linewidth=0.5, rasterized=True, label="$t=100\ \mathrm{a}$")
        ax.set_xlim(0, 640e3)
        ax.set_ylim(0, 40e3)
        ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
        ax.set_yticks([0, 40e3])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax_text(ax, fig, f"{subplot_labels[i]}: {labels[field]}", dy=0.27, top=True)
        ax_text(ax, fig, f"$N_v = {msh.num_vertices()}$", top=False)

        # plot the bounding box around the zoomed-in region
        ax.plot([380e3, 520e3, 520e3, 380e3, 380e3], [0, 0, 40e3, 40e3, 0], '--',
                color="C0", linewidth=0.5, rasterized=True,
                path_effects=[pe.Stroke(linewidth=1.0, foreground='black'), pe.Normal()]
                )

        # Plot the zoomed-in region
        ax_zoom = axes[i, 1]
        triplot(msh, axes=ax_zoom, interior_kw=interior_kw)
        ax_zoom.plot(ref4_gl_90[:, 0], ref4_gl_90[:, 1], 'y--', linewidth=0.5, rasterized=True)
        ax_zoom.plot(ref4_gl_100[:, 0], ref4_gl_100[:, 1], 'y-', linewidth=0.5, rasterized=True)
        ax_zoom.set_xlim(380e3, 520e3)
        ax_zoom.set_ylim(0, 40e3)
        ax_zoom.set_xticks([400e3, 500e3])
        ax_zoom.set_yticks([0, 40e3])
        ax_zoom.set_xticklabels([])
        ax_zoom.set_yticklabels([])

    # Set labels and legend for the last row
    ax = axes[-1, 0]
    ax.set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-6)
    ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-13)
    ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
    ax.set_xticklabels(["$0$"] + [""]*6 + ["$640$"])
    ax.set_yticklabels(["$-40$", "$0$"])
    ax.xaxis.set_tick_params(pad=1)
    ax.yaxis.set_tick_params(pad=1)

    ax_zoom = axes[-1, 1]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[4:], labels=labels[4:],
        loc='upper center', bbox_to_anchor=(0.84, -0.25),
        title="Reference grounding line contours",
        ncol=2,
    )

    return fig

def plot_hessian_aspect_ratio(args):
    from animate.quality import QualityMeasure

    c = 6400
    fields = ["h", "u", "u-int-h", "tau"]
    labels = {"h": "$H$", "tau": r"$\tau_b$", "u": r"$\mathbf{u}$", "u-int-h": r"$(H,\,\mathbf{u})$"}
    subplot_labels = ["(a)", "(b)", "(c)", "(d)",]

    gridspec_kw = {"hspace": 0.05, "height_ratios": [0.2] + [1] * len(fields)}
    fig, axes = plt.subplots(len(fields)+1, 1, figsize=(3.27, 3), gridspec_kw=gridspec_kw)

    for i, field in enumerate(fields):
        simulation_id = f"{c}_{field}_20_hybrid"
        fpath = os.path.join(args.output_dir, simulation_id, f"outputs-Ice1-id_{simulation_id}.h5")
        with CheckpointFile(fpath, "r") as afile:
            msh = afile.load_mesh(f"mesh_iter_4_int_9")
        qm = QualityMeasure(msh)
        ar = qm("aspect_ratio")

        ax = axes[i+1]
        im = tripcolor(ar, axes=ax, cmap=cmap_devon_reversed, rasterized=True, norm=LogNorm(vmin=1e0, vmax=1e2))
        ax.set_xlim(0, 640e3)
        ax.set_ylim(0, 40e3)
        ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
        ax.set_yticks([0, 40e3])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax_text(ax, fig, f"{subplot_labels[i]}: {labels[field]}", dy=0.28, top=True, colour="white")

    ax = axes[-1]
    ax.set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-8)
    ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-12)
    ax.set_xticks([jj*100e3 for jj in range(7)] + [640e3])
    ax.set_xticklabels(["$0$"] + [""]*6 + ["$640$"])
    ax.set_yticklabels(["$-40$", "$0$"])

    cax = axes[0]
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', pad=0.)
    cax_pos = cax.get_position()
    cax_pos.x0 += 0.12
    cax_pos.x1 -= 0.12
    cax_pos.y0 += 0.04
    cax_pos.y1 += 0.04
    cax.set_position(cax_pos)
    cbar.ax.xaxis.set_tick_params(pad=0, labelsize=6)
    cbar.ax.set_title(r"Aspect ratio of adapted mesh elements of $\mathcal{H}_9$", pad=0.2)

    return fig

def plot_hessian_err_spatial(args):
    c = 6400

    gridspec_kw = {"height_ratios": [1, 1, 1, 1, 1, 1, 0.5]}
    fig, axes = plt.subplots(7, 1, figsize=(3.27, 4.0), gridspec_kw=gridspec_kw)

    fields = ["h", "u", "tau", "u-int-h"]
    labels = {"h": "$H$", "tau": r"$\tau_b$", "u": r"$\mathbf{u}$", "u-int-h": r"$(H,\,\mathbf{u})$"}

    ref_fpath = os.path.join(args.output_dir, "uniform_250", "outputs-Ice1-id_uniform_250.h5")
    with CheckpointFile(ref_fpath, "r") as afile:
        ref_msh = afile.load_mesh("mesh_iter_0_int_0")
        ref_h_100a = afile.load_function(ref_msh, "thickness_iter_0_int_0_yr_100.0")
        ref_u_100a = afile.load_function(ref_msh, "velocity_iter_0_int_0_yr_100.0")
    ref_umag = sqrt(dot(ref_u_100a, ref_u_100a))
    Q = FunctionSpace(ref_msh, "CG", 1)
    V = VectorFunctionSpace(ref_msh, "CG", 1)

    for i, field in enumerate(fields):
        sim_dir_path = os.path.join(args.output_dir, f"{c}_{field}_20_hybrid/")
        fpath = os.path.join(sim_dir_path, f"outputs-Ice1-id_{c}_{field}_20.h5")

        with CheckpointFile(fpath, "r") as afile:
            msh = afile.load_mesh(f"mesh_iter_4_int_9")
            h_100a = afile.load_function(msh, f"thickness_iter_4_int_9_yr_100.0")
            u_100a = afile.load_function(msh, f"velocity_iter_4_int_9_yr_100.0")

        proj_h = project(h_100a, Q)
        proj_u = project(u_100a, V)
        proj_umag = sqrt(dot(proj_u, proj_u))

        h_norm = norm(proj_h - ref_h_100a) / norm(ref_h_100a)
        u_norm = norm(proj_umag - ref_umag) / norm(ref_umag)

        print(h_norm, u_norm)

        rel_diff_h = Function(Q).interpolate(abs(proj_h - ref_h_100a) / abs(ref_h_100a))
        rel_diff_umag = Function(Q).interpolate(abs(proj_umag - ref_umag) / abs(ref_umag))

        tripcolor_kw = {"cmap": cmap_vik, "norm": LogNorm(vmin=1e-3, vmax=1e0), "rasterized": True}
        bbox_kw = {"boxstyle": "square,pad=0.1", "facecolor": "white", "alpha": 0.5, "edgecolor": "none"}
        dx_label_pos, dy_label_pos = 30e3, 0.9e3

        ax_h = axes[i * 2]
        im = tripcolor(rel_diff_h, axes=ax_h, **tripcolor_kw)
        x_label_pos, y_label_pos = get_label_pos(ax_h, fig, top=False)
        x_label_pos += dx_label_pos
        y_label_pos += dy_label_pos
        t = ax_h.text(x_label_pos, y_label_pos, r"$\tilde{e}_H=$" + f" ${h_norm:.3f}$", bbox=bbox_kw)

        ax_u = axes[i * 2 + 1]
        im = tripcolor(rel_diff_umag, axes=ax_u, **tripcolor_kw)
        x_label_pos, y_label_pos = get_label_pos(ax_u, fig, top=False)
        x_label_pos += dx_label_pos
        y_label_pos += dy_label_pos
        t = ax_u.text(x_label_pos, y_label_pos, r"$\tilde{e}_{\mathbf{u}}=$" + f" ${u_norm:.3f}$", bbox=bbox_kw)

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 640e3)
        ax.set_ylim(0, 40e3)
        ax.set_xticks([jj * 100e3 for jj in range(7)] + [640e3])
        ax.set_yticks([0, 40e3])
        ax.xaxis.set_tick_params(pad=0.5)
        ax.yaxis.set_tick_params(pad=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax = axes[-2]
    ax.set_xlabel(r"$x\ (\mathrm{km})$", labelpad=-6)
    ax.set_ylabel(r"$y\ (\mathrm{km})$", labelpad=-10)
    ax.set_xticklabels(["$0$"] + [""] * 6 + ["$640$"])
    ax.set_yticklabels(["-$40$", "$0$"])

    for i in range(3):
        ax = axes[i * 2]
        x_label_pos, y_label_pos = get_label_pos(ax, fig, top=True)
        y_label_pos -= 2e3
        letter_label = ["(c)", "(d)", "(e)"][i]
        label = rf"{letter_label}: {labels[fields[i]]}"
        ax.text(x_label_pos, y_label_pos, label, color="white")

    # Remove hspace between axes
    for i in range(3):
        pos0 = axes[i * 2].get_position()
        pos1 = axes[i * 2 + 1].get_position()
        diff_y = pos0.y0 - pos1.y1
        pos0.y0 -= diff_y / 2
        pos1.y1 += diff_y / 2
        axes[i * 2].set_position(pos0)
        axes[i * 2 + 1].set_position(pos1)

    # add colorbar to the last axis
    axes[-1].axis("off")
    cbar_ax = fig.add_axes([0.16, 0.08, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=1.0, aspect=60)
    for c in cbar.ax.collections:
        c.set_rasterized(True)
    cbar_title = r"$\left|f_{\mathrm{int}} - f_{\mathrm{ref}}\right|\,/f_{\mathrm{ref}},\ f=H,\,\left|\mathbf{u}\right|$"
    cbar.set_label(cbar_title)
    cbar.ax.xaxis.set_label_position('top')

    return fig

def plot_hessian_err_avg(args):
    categorical_colors = cmap_devonS(np.linspace(0, 1, 11))
    cs = [800 * 2**i for i in range(5)]

    gridspec_kw = {"height_ratios": [1, 1, 0.2]}
    # 3.27
    fig, axes = plt.subplots(3, 1, figsize=(3.27, 4.0), gridspec_kw=gridspec_kw)

    fields = ["h", "u", "tau", "u-int-h"]
    labels = {"h": "$H$", "tau": r"$\tau_b$", "u": r"$\mathbf{u}$", "u-int-h": r"$(H,\,\mathbf{u})$"}
    markers = {"uniform": ".", "h": "+", "u": "s", "tau": "x", "u-int-h": "o"}
    linestyles = {"uniform": ":", "h": "-", "u": "-", "tau": "-", "u-int-h": "-"}

    for i, field in enumerate(fields):
        d = []
        n_h = []
        n_umag = []
        for c in cs:
            sim_dir_path = os.path.join(args.output_dir, f"{c}_{field}_20_hybrid/")
            dofs = np.load(os.path.join(sim_dir_path, "dofs.npy"))
            norms = np.load(os.path.join(sim_dir_path, "norms.npy"))
            d.append(np.mean(dofs, axis=1)[-1])
            norms_avg_time = np.mean(norms, axis=2)
            n_h.append(norms_avg_time[0, 0])
            n_umag.append(norms_avg_time[1, 0])

        axes[0].loglog(d, n_h, label=labels[field], marker=markers[field], fillstyle='none', linestyle=linestyles[field])
        axes[1].loglog(d, n_umag, label=labels[field], marker=markers[field], fillstyle='none', linestyle=linestyles[field])

    field = "uniform"
    ref_d, ref_n_h, ref_n_umag = [], [], []
    for ref_lv in range(4):
        fpath = os.path.join(args.output_dir, f"uniform_{250 * 2**ref_lv}", "analysis", "norms.npy")
        norms = np.load(fpath)
        ref_d.append(np.load(fpath.replace("norms", "dofs"))[0, 0])
        norms_avg_time = np.mean(norms, axis=1)
        ref_n_h.append(norms_avg_time[0])
        ref_n_umag.append(norms_avg_time[1])
    axes[0].loglog(ref_d, ref_n_h, label="Uniform", marker=markers[field], color='k', linestyle=linestyles[field])
    axes[1].loglog(ref_d, ref_n_umag, label="Uniform", marker=markers[field], color='k', linestyle=linestyles[field])

    for i, ax in enumerate(axes[:2]):
        ax.set_xlim(1e3, 1.2e5)
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.grid(which='both', linestyle=':', linewidth='0.5', color='black')
        label = ["(a)", "(b)"][i]
        ax_text(ax, fig, label, d=0.04, top=False)
    axes[0].set_xticklabels([])
    axes[0].set_ylim(1e-3, 2e-1)
    axes[1].set_ylim(1e-2, 1e-0)
    axes[1].set_xlabel("Average $N_v$")
    axes[0].set_ylabel(r"Average $\tilde{e}_H$")
    axes[1].set_ylabel(r"Average $\tilde{e}_{\mathbf{u}}$")
    axes[1].set_xticklabels([r"$10^3$", r"$10^4$", r"$10^5$"])

    # move axes[1] slightly up
    pos = axes[1].get_position()
    pos.y0 += 0.025
    pos.y1 += 0.025
    axes[1].set_position(pos)

    handles, labels = axes[0].get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[3], handles[2], handles[4]]
    labels = [labels[0], labels[1], labels[3], labels[2], labels[4]]
    legend_ax = axes[2]
    legend_ax.legend(handles, labels, ncols=5, frameon=False)
    pos = legend_ax.get_position()
    pos.y0 -= 0.01
    pos.y1 -= 0.01
    legend_ax.set_position(pos)
    legend_ax.axis("off")

    return fig

def plot_hessian_cpu_time_error(args):
    cs = [800, 3200, 12800]
    fields = ["h", "u", "tau", "u-int-h",]
    labels = ["$H$", r"$\mathbf{u}$", r"$\tau_b$", r"$(H,\mathbf{u})$", "Uniform"]
    markers = {"h": "+", "u": "s", "tau": "x", "u-int-h": "o", "uniform": "."}

    cpu_times = {field: {c: None for c in cs} for field in fields}
    h_errors = {field: {c: None for c in cs} for field in fields}
    u_errors = {field: {c: None for c in cs} for field in fields}

    for i, field in enumerate(fields):
        d, n_h, n_umag = [], [], []
        for c in cs:
            analysis_path = os.path.join(args.output_dir, f"{c}_{field}_20_hybrid", "analysis")
            norms = np.load(os.path.join(analysis_path, "norms.npy"))
            norms_avg_time = np.mean(norms, axis=2)
            h_errors[field][c] = norms_avg_time[0, -1]
            u_errors[field][c] = norms_avg_time[1, -1]

    # put in uniform resolution separately
    uniform_h_errors, uniform_u_errors = [], []
    for res in 4000, 2000, 1000, 500:
        fpath = os.path.join(args.output_dir, f"uniform_{res}", "analysis", "norms.npy")
        norms = np.load(fpath)
        norms_avg_time = np.mean(norms, axis=1)
        uniform_h_errors.append(norms_avg_time[0])
        uniform_u_errors.append(norms_avg_time[1])
    cpu_times["uniform"] = [2.7, 9.3, 36.8, 200]  # manually read from time.txt
    h_errors["uniform"] = uniform_h_errors
    u_errors["uniform"] = uniform_u_errors

    time_csv_fpath = os.path.join(args.output_dir, "time.txt")
    with open(time_csv_fpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            simulation_id, cpu_time = line.split(",")
            _, c, f, _, _ = simulation_id.split("_")
            cpu_times[f][int(c)] = float(cpu_time)

    # Plotting the error vs CPU time for each field
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.27, 2.7))
    
    for ax, err in zip([ax1, ax2], [h_errors, u_errors]):
        for field in fields:
            ax.loglog(cpu_times[field].values(), err[field].values(), marker=markers[field], label=labels[fields.index(field)], fillstyle='none')
        ax.loglog(cpu_times["uniform"], err["uniform"], marker=markers["uniform"], label="Uniform", color="k", linestyle=":")

    ax1.set_ylabel(r"Average $\tilde{e}_H$")
    ax1.set_xticklabels([])
    ax2.set_ylabel(r"Average $\tilde{e}_{\mathbf{u}}$")
    ax2.set_xlabel(r"CPU time ($\mathrm{min}$)")

    ax1.set_ylim(1e-3, 2e-1)
    ax2.set_ylim(1e-2, 1e-0)
    for ax in [ax1, ax2]:
        ax.grid(which='both', linestyle=':', linewidth='0.5', color='black')
        ax.set_xlim(1e0, 2.2e2)

    for i, ax in enumerate([ax1, ax2]):
        ax_text(ax, fig, ["(a)", "(b)"][i], d=0.04, dy=0.18, top=True)

    ax2_pos = ax2.get_position()
    ax2_pos.y0 += 0.035
    ax2_pos.y1 += 0.035
    ax2.set_position(ax2_pos)

    # create a new axis for legend
    legend_ax = fig.add_axes([0.1, -0.03, 0.5, 0.5], frameon=False)
    legend_ax.axis('off')

    handles, labels = ax1.get_legend_handles_labels()
    _handles = [handles[0], handles[1], handles[3], handles[2], handles[4]]
    _labels = [labels[0], labels[1], labels[3], labels[2], labels[4]]
    legend_ax.legend(_handles, _labels, loc="lower left", ncol=5)

    return fig

def plot_hessian_cpu_time_contribution(args):
    def extract_times_from_flamegraph(file_path):
        solver_lines = [
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.fixed_point_iteration;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.variational_solver.",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.fixed_point_iteration;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.function.Function.interpolate",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.fixed_point_iteration;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.function.Function.assign",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.fixed_point_iteration;goalie.mesh_seq.MeshSeq.solve_forward;LoadMesh",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.fixed_point_iteration;goalie.mesh_seq.MeshSeq.solve_forward;LoadFunction",

            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;firedrake.variational_solver.",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;firedrake.function.Function.interpolate",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;firedrake.function.Function.assign",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;LoadMesh",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;LoadFunction",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;firedrake.function.Function.copy",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.on_the_fly;firedrake.functionspace.VectorFunctionSpace",

            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.variational_solver",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.function.Function.interpolate",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.function.Function.assign",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;LoadMesh",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;LoadFunction",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.function.Function.copy",
            "Stage;firedrake;goalie.mesh_seq.MeshSeq.solve_forward;firedrake.functionspace.VectorFunctionSpace",

            # "Stage;firedrake;goalie.mesh_seq.MeshSeq.__init__",
        ]
        adapt_lines = [
            "fpi_adaptor",
            "fly_adaptor"
        ]
        transfer_lines = [
            "animate.interpolation.transfer"
        ]

        with open(file_path, 'r') as file:
            lines = file.readlines()

        total_time = 0
        pde_solve_time = 0
        mesh_adaptation_time = 0
        transfer_time = 0

        for line in lines:
            *fn, time = line.split(" ")
            time = int(time)
            total_time += time

            for adapt_line in adapt_lines:
                if adapt_line in fn[1]:
                    mesh_adaptation_time += time
            for solver_line in solver_lines:
                if solver_line in fn[1]:
                    pde_solve_time += time
            for transfer_line in transfer_lines:
                if transfer_line in fn[1]:
                    transfer_time += time

        pde_solve_time = pde_solve_time / total_time * 100
        mesh_adaptation_time = mesh_adaptation_time / total_time * 100
        transfer_time = transfer_time / total_time * 100

        return pde_solve_time, mesh_adaptation_time, transfer_time

    fig, ax3 = plt.subplots(figsize=(3.27, 1.25))

    cs = [800, 3200, 12800]
    fields = ["h", "u", "tau", "u-int-h"]

    cpu_times_percent = {c: {f: None for f in fields} for c in cs}
    _fields = ["h", "u", "u-int-h", "tau"]
    for c in cs:
        for f in _fields:
            flame_fpath = os.path.join(args.output_dir, f"flamegraph_{c}_{f}_20.txt")
            pde_solve_time, mesh_adaptation_time, transfer_time = extract_times_from_flamegraph(flame_fpath)
            cpu_times_percent[c][f] = [pde_solve_time, mesh_adaptation_time, transfer_time]

    bar_width = 0.2
    bar_colors = ['blue', 'orange', 'green']
    components = ["PDE solve", "Mesh adaptation", "Interpolation"]
    group_spacing = 0.4
    bar_spacing = 0.05

    bar_positions = []
    group_start_positions = []

    # Calculate bar positions with separation
    for i, c in enumerate(cs):
        group_start = 0.3 + i * (len(fields) * (bar_width + bar_spacing) + group_spacing)
        group_start_positions.append(group_start + (len(fields) * bar_width + (len(fields) - 1) * bar_spacing) / 2)
        for j, f in enumerate(fields):
            bar_positions.append(group_start + j * (bar_width + bar_spacing))

    bottoms = np.zeros(len(bar_positions))

    # Create grouped, stacked bars
    for i, (comp, color) in enumerate(zip(components, bar_colors)):
        values = []
        for c in cs:
            for f in _fields:
                values.append(cpu_times_percent[c][f][i])
        ax3.bar(bar_positions, values, bar_width, bottom=bottoms, label=comp, color=color)
        bottoms += values

    # Set primary x-axis for bar positions
    ax3.set_xticks(bar_positions)
    _labels = ["$H$", r"$\mathbf{u}$", r"$(H,\!\mathbf{u})$", r"$\tau_b$"]
    ax3.set_xticklabels(_labels * len(cs), rotation=90)

    # Add secondary x-axis for grouping (manually placing text labels)
    for i, group_start in enumerate(group_start_positions):
        group_start -= 0.1
        ax3.text(group_start, -0.4, rf"$\mathcal{{C}}={cs[i]}$", ha='center', va='center', transform=ax3.get_xaxis_transform())

    # Final adjustments
    ax3.set_ylabel("CPU time (%)")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.45, -0.45), ncol=3)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.set_ylim(0, 100)
    total_width = (len(fields) * (bar_width + bar_spacing) * len(cs)) + (group_spacing * (len(cs) - 1))
    ax3.set_xlim(-0.2, total_width + 0.5)

    # align labels
    fig.align_labels()

    return fig

###############################################################

def main():
    fig_fun_map = {
        "initial_steady_state": plot_initial_steady_state,
        "uniform_convergence": plot_uniform_convergence,
        "metric_components": plot_metric_components,
        "metric_intersection": plot_metric_intersection,
        "strat_comparison_meshes": plot_strat_comparison_meshes,
        "strat_comparison_conv": plot_strat_comparison_conv,
        "strat_comparison_dofs": plot_strat_comparison_dofs,
        "hessian_meshes": plot_hessian_meshes,
        "hessian_aspect_ratio": plot_hessian_aspect_ratio,
        "hessian_err_spatial": plot_hessian_err_spatial,
        "hessian_err_avg": plot_hessian_err_avg,
        "hessian_cpu_time_error": plot_hessian_cpu_time_error,
        "hessian_cpu_time_contribution": plot_hessian_cpu_time_contribution,
    }
    fig_names = list(fig_fun_map.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,)
    parser.add_argument("--fig", type=str, required=True, choices=fig_names)
    parser.add_argument("--format", type=str, default="pdf")
    args = parser.parse_args()

    fig = fig_fun_map[args.fig](args)

    # create figures dir if it does not exist
    if not os.path.exists(os.path.join(args.output_dir, "figures")):
        os.makedirs(os.path.join(args.output_dir, "figures"))

    fig_idx = fig_names.index(args.fig) + 1
    fname = f"fig{fig_idx}.{args.format}"
    fig.savefig(os.path.join(args.output_dir, "figures", fname),
                # backend="pgf",
                dpi=300, bbox_inches='tight', pad_inches=0.01, transparent=True)
    print(f"Saved {fname}.")

if __name__ == "__main__":
    main()
