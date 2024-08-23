import firedrake
from firedrake.pyplot import *
import numpy as np
import os
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys

from utility_functions import *

get_analysis_dir = lambda simulation_id: os.path.join(
    output_dir_path, simulation_id, "analysis"
)

# lv is the level of refinement (0, 1, 2, 3, 4, where 0 is 4000m and 4 is 250m)
get_ref_scatter_kwargs = lambda lv: {
    "s": 25,
    "alpha": 0.8,
    "label": f"{4/2**lv} km",
    "facecolors": "none",
    "edgecolors": f"C{lv}",
}


def base_time_plot(ax):
    ax.set(
        xlabel="Years",
        xlim=(0, 200),
    )
    ax.grid(alpha=0.3)


def mass_time_plot(ax):
    base_time_plot(ax)
    ax.set(
        ylabel=r"Ice mass (Gt)",
        ylim=(48, 56),
        title="Ice mass vs time",
    )


def midline_xgl_time_plot(ax):
    base_time_plot(ax)
    ax.set(
        ylabel="Midline xgl",
        ylim=(3.8e2, 4.6e2),
        title="Midline xgl vs time",
    )


def dofs_time_plot(ax):
    base_time_plot(ax)
    ax.set(
        ylabel="Degrees of freedom",
        # ylim=(1e4, 1e6),
        title="Degrees of freedom vs time",
    )


def convergence_time_plot(ax):
    base_time_plot(ax)
    ax.set(
        ylabel="Convergence",
        title="Convergence vs time",
    )


def gl_plot(ax, year):
    ax.set(
        xlabel="x (km)",
        ylabel="y (km)",
        xlim=(3.8e2, 5.2e2),
        ylim=(0, 4e1),
        title=f"Grounding line at year {year}",
    )
    ax.grid(alpha=0.3)


def resolution_gl_plot(ax):
    ax.set(
        xlabel="y (m)",
        ylabel="Resolution at gl (m)",
        # xlim=(0, 4e4),
        # title='Grounding line at year 100',
    )


def mesh_plot(ax, msh, ax_params={}):
    from firedrake import Function, FunctionSpace

    ax.set(**ax_params)
    tripcolor_kwargs = dict(
        axes=ax,
        cmap="coolwarm",
        norm=LogNorm(),
        shading="flat",
    )
    interior_kwargs = dict(
        linewidth=0.08,
    )

    cell_sizes = Function(FunctionSpace(msh, "DG", 0)).interpolate(msh.cell_sizes)
    im = tripcolor(cell_sizes, **tripcolor_kwargs)
    triplot(msh, axes=ax, interior_kw=interior_kwargs)

    return im


def indicators_plot(ax, ind):
    ax_params = dict(
        ylabel="y (km)",
        xlim=(0, 6.4e5),
        ylim=(0, 4e4),
    )
    ax.set(**ax_params)
    tricontourf_kwargs = dict(
        axes=ax,
        cmap="coolwarm",
        norm=LogNorm(),
        shading="flat",
    )

    im = tricontourf(ind, **tricontourf_kwargs)

    return im


def mesh_evolution_plot(axes, fs, gls=None, title=None):
    ax_params = dict(
        ylabel="y (km)",
        xlim=(0, 6.4e2),
        ylim=(0, 4e1),
    )

    if gls is None:
        gls = [None] * len(fs)

    for ax, f, gl in zip(axes, fs, gls):
        year = f.name().split("_")[-1]
        msh = f.function_space().mesh()
        msh.coordinates.dat.data[:] /= 1e3  # Convert to km
        im = mesh_plot(ax, msh, ax_params)

        if gl is not None:
            ax.plot(gl[:, 0] / 1e3, gl[:, 1] / 1e3, "y", label="Grounding line")

        # text for year in upper left corner
        ax.text(
            0.05,
            0.95,
            f"Year {year}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

    axes[-1].set_xlabel("x (km)")
    fig = axes[0].get_figure()
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax)


def plot_mesh_evolution(simulation_id, iter):
    years = [50.0, 100.0, 150.0, 200.0]
    years = [year - 1 for year in years]  # convert to index (year 0 isn't saved)

    analysis_dir = get_analysis_dir(simulation_id)
    simulation_fpath = os.path.join(
        os.path.dirname(analysis_dir), f"outputs-Ice1-id_{simulation_id}.h5"
    )

    # hs, _ = get_adapted(simulation_fpath, iterations=iter, years=years)
    # for on the fly
    hs, _ = get_adapted(simulation_fpath, iterations=1, years=years)

    gls = np.load(os.path.join(analysis_dir, "gls.npy"))[iter]
    gls = [gls[int(year)] for year in years]

    fig, axes = plt.subplots(4, figsize=(10, 20), sharex=True)
    mesh_evolution_plot(axes, hs, gls)

    # plot 250m uniform reference values
    ref_dir = os.path.join(output_dir_path, "uniform_250", "analysis")
    ref_gls = np.load(os.path.join(ref_dir, "gls.npy"))[0, :, ::10]
    scatter_kw = get_ref_scatter_kwargs(4)

    for ax, year in zip(axes, years):
        ax.scatter(
            ref_gls[int(year), :, 0] / 1e3, ref_gls[int(year), :, 1] / 1e3, **scatter_kw
        )

    # fig.suptitle(simulation_id)
    fig.savefig(
        os.path.join(analysis_dir, "mesh_evolution.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def summarise_output(
    simulation_id, ref_lv4_masses, ref_lv4_midline_x_gls, masses_all, midline_x_gls_all
):
    # nqa - do not use for analysis, just for fast visualisation
    
    ref_lv4_masses = ref_lv4_masses[1:]
    ref_lv4_midline_x_gls = ref_lv4_midline_x_gls[1:]

    rel_diffs_masses = [
        np.abs(masses - ref_lv4_masses) / ref_lv4_masses for masses in masses_all
    ]
    rel_diffs_midline_x_gls = [
        np.abs(midline_x_gls - ref_lv4_midline_x_gls) / ref_lv4_midline_x_gls
        for midline_x_gls in midline_x_gls_all
    ]
    sum_rel_diffs = np.array(rel_diffs_masses) + np.array(rel_diffs_midline_x_gls)

    mean_rel_diffs_masses = np.mean(rel_diffs_masses, axis=1)
    best_iter_masses = np.argmin(mean_rel_diffs_masses)
    mean_rel_diffs_midline_x_gls = np.nanmean(rel_diffs_midline_x_gls, axis=1)
    best_iter_midline_x_gls = np.argmin(mean_rel_diffs_midline_x_gls)
    mean_sum_rel_diffs = np.nanmean(sum_rel_diffs, axis=1)
    best_iter_both = np.argmin(mean_sum_rel_diffs)

    # write the best iteration to simulation_summary.csv
    # first check if the file exists, if not, create it and write a header
    header = (
        "simulation_id,"
        "adapt_field,"
        "num_subintervals,"
        "num_iterations,"
        "best_iter_masses,"
        "best_rel_diffs_mass,"
        "best_iter_midline_x_gls,"
        "best_rel_diffs_midline_x_gls,"
        "best_iter_both,"
        "best_sum_rel_diffs_both\n"
    )
    simulation_summary_csv_path = os.path.join(output_dir_path, "simulation_summary.csv")
    if not os.path.exists(simulation_summary_csv_path):
        with open(simulation_summary_csv_path, "w") as f:
            f.write(header)

    # check if simulation_id is in the csv file already and remove it if it is
    with open(simulation_summary_csv_path, "r") as f:
        lines = f.readlines()
    with open(simulation_summary_csv_path, "w") as f:
        f.write(header)
        for line in lines[1:]:
            if simulation_id not in line:
                f.write(line)
    id_args = simulation_id.split("_")
    complexity, adapt_field, num_subintervals = id_args

    summary = (
        f"{simulation_id},"
        f"{adapt_field},"
        f"{num_subintervals},"
        f"{len(masses_all)},"  # num_iterations
        f"{best_iter_masses},"
        f"{mean_rel_diffs_masses[best_iter_masses]:.5f},"
        f"{best_iter_midline_x_gls},"
        f"{mean_rel_diffs_midline_x_gls[best_iter_midline_x_gls]:.5f},"
        f"{best_iter_both},"
        f"{mean_sum_rel_diffs[best_iter_both]:.5f}\n"
    )
    with open(simulation_summary_csv_path, "a") as f:
        f.write(summary)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for ax in axes:
        convergence_time_plot(ax)
    for iter in range(len(masses_all)):
        axes[0].plot(
            np.arange(1.0, 201.0, 1.0), rel_diffs_masses[iter], label=f"iter {iter}"
        )
        axes[1].plot(
            np.arange(1.0, 201.0, 1.0),
            rel_diffs_midline_x_gls[iter],
            label=f"iter {iter}",
        )
        axes[2].plot(
            np.arange(1.0, 201.0, 1.0), sum_rel_diffs[iter], label=f"iter {iter}"
        )
    axes[0].legend(loc="upper right")
    axes[0].set_title("Masses")
    axes[1].set_title("Midline x gls")
    axes[2].set_title("Sum of both")
    fig.savefig(
        os.path.join(get_analysis_dir(simulation_id), "convergence.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return best_iter_both


def plot_output(simulation_id):
    def _plot_data(ax, data_all, plot_func, *plot_args):
        plot_func(ax, *plot_args)
        for i, data in enumerate(data_all):
            ax.plot(*data, label=f"iter {i}")

    analysis_dir = get_analysis_dir(simulation_id)
    masses_all = np.load(os.path.join(analysis_dir, "masses.npy"))
    vafs_all = np.load(os.path.join(analysis_dir, 'vafs.npy'))
    for vafs in vafs_all:
        vafs -= vafs[0]
    gls_all = np.load(os.path.join(analysis_dir, "gls.npy"))
    midline_x_gls_all = np.load(os.path.join(analysis_dir, "midline_x_gls.npy"))
    dofs_all = np.load(os.path.join(analysis_dir, "dofs.npy"))
    # resolution_at_gls_yr100 = np.load(
    #     os.path.join(analysis_dir, "resolution_at_gls_yr100.npy")
    # )

    k = int(200 / len(masses_all[0]))
    years = np.flip(np.arange(200., 0., -k))

    # years = np.arange(0.0, 201.0, 1.0)
    mid_idx = 100  # index of the year 100

    fig, axes = plt.subplots(6, figsize=(10, 30))
    _plot_data(axes[0], [(years, masses) for masses in masses_all], mass_time_plot)
    _plot_data(
        axes[1],
        [(years, midline_x_gls / 1e3) for midline_x_gls in midline_x_gls_all],
        midline_xgl_time_plot,
    )
    _plot_data(
        axes[2],
        [(gls[mid_idx, :, 0] / 1e3, gls[mid_idx, :, 1] / 1e3) for gls in gls_all],
        gl_plot,
        years[mid_idx],
    )  # year 100
    _plot_data(
        axes[3],
        [(gls[-1, :, 0] / 1e3, gls[-1, :, 1] / 1e3) for gls in gls_all],
        gl_plot,
        years[-1],
    )  # year 200
    axes[4].axhline(1771, color="k", linestyle="--", alpha=0.4)
    _plot_data(axes[4], [(years, dofs) for dofs in dofs_all], dofs_time_plot)
    # set legend labels to be the mean dofs for each line
    axes[4].legend(
        ["DoFs = 1771"] +
        [f"iter {i} - {np.mean(dofs):.0f}" for i, dofs in enumerate(dofs_all)],
        loc="upper right",
    )

    _plot_data(axes[5], [(years, vafs) for vafs in vafs_all], dofs_time_plot)

    # plot reference values
    years = np.arange(1., 201., 1.)
    for lv in [0, 1, 2, 3, 4]:
        res = int(4000 / 2 ** lv)
        ref_dir = os.path.join(output_dir_path, f"uniform_{res}", "analysis")
        plot_every_n_yrs = 10

        ref_masses = np.load(os.path.join(ref_dir, "masses.npy"))[0]
        ref_vafs = np.load(os.path.join(ref_dir, "vafs.npy"))[0]
        ref_vafs -= ref_vafs[0]
        ref_gls = np.load(os.path.join(ref_dir, "gls.npy"))[0]
        ref_midline_x_gls = np.load(os.path.join(ref_dir, "midline_x_gls.npy"))[0]

        scatter_kw = get_ref_scatter_kwargs(lv)

        print()
        axes[0].scatter(
            years[::plot_every_n_yrs], ref_masses[::plot_every_n_yrs], **scatter_kw
        )
        axes[1].scatter(
            years[::plot_every_n_yrs],
            ref_midline_x_gls[::plot_every_n_yrs] / 1e3,
            **scatter_kw,
        )
        axes[2].scatter(
            ref_gls[mid_idx, :, 0] / 1e3, ref_gls[mid_idx, :, 1] / 1e3, **scatter_kw
        )
        axes[3].scatter(ref_gls[-1, :, 0] / 1e3, ref_gls[-1, :, 1] / 1e3, **scatter_kw)

        # vafs
        axes[5].scatter(
            years[::plot_every_n_yrs], ref_vafs[::plot_every_n_yrs], **scatter_kw
        )

    axes[5].set_title("Change in VAF")

    axes[0].legend(loc="upper right")
    fig.savefig(os.path.join(analysis_dir, "plot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return ref_masses, ref_midline_x_gls, masses_all, midline_x_gls_all


def analyse_output(simulation_id, repeat_analysis, k=None):
    # directory, file_name = os.path.split(fpath)
    simulation_output_fpath = os.path.join(
        output_dir_path, simulation_id, f"outputs-Ice1-id_{simulation_id}.h5"
    )
    analysis_dir = get_analysis_dir(simulation_id)

    if os.path.exists(analysis_dir):
        # check if masses.npy, gls.npy and midline_x_gls.npy exist
        files = [
            os.path.join(analysis_dir, f"{f}.npy")
            for f in [
                "masses",
                "vafs",
                "gls",
                "midline_x_gls",
                "dofs",
            ]
        ]
        if all([os.path.exists(f) for f in files]) and not repeat_analysis:
            print(f"Analysis already done for {simulation_id}.")
            return True
    else:
        os.makedirs(analysis_dir)

    if 'uniform' in simulation_id:
        hs_all = []
        us_all = []
        simulation_output_fpath = os.path.join(output_dir_path, simulation_id, f"outputs-Ice1-id_{simulation_id}.h5")
        with CheckpointFile(simulation_output_fpath, "r") as afile:
            # for i in range(2):
            msh = afile.load_mesh(f"mesh_iter_0_int_0")
            for yr in np.arange(1., 201., 1.):
                h = afile.load_function(msh, name=f"thickness_iter_0_int_0_yr_{yr}")
                u = afile.load_function(msh, name=f"velocity_iter_0_int_0_yr_{yr}")
                hs_all.append(h)
                us_all.append(u)
    else:
        years = np.flip(np.arange(200., 1., -k)) if k else None
        hs_all, us_all = get_adapted(simulation_output_fpath, iterations="all", years=years)
    # check if hs_all is one-dimensional
    if len(np.shape(hs_all)) == 1:
        # Expand dimension if it is one dimensional
        hs_all = np.expand_dims(hs_all, axis=0)
        us_all = np.expand_dims(us_all, axis=0)

    masses_all, vafs_all, gls_all, midline_x_gls_all = [], [], [], []
    for iter, (hs, us) in enumerate(zip(hs_all, us_all)):
        masses_all.append(get_mass(hs))
        hafs = get_haf(hs)
        vafs = get_vaf(hafs)
        vafs_all.append(vafs)
        gls = get_gl(hafs)
        gls_all.append(gls)
        midline_x_gls_all.append(get_midline_x_gl(gls))

    longest_gl_length = 0
    for gls in gls_all:
        for gl in gls:
            gl_length = len(gl)
            longest_gl_length = (
                gl_length if gl_length > longest_gl_length else longest_gl_length
            )

    new_gls_all = [[[] for _ in gls] for gls in gls_all]
    for i, gls in enumerate(gls_all):
        for j, gl in enumerate(gls):
            gl_length = len(gl)
            if gl_length < longest_gl_length:
                rows_to_add = longest_gl_length - gl_length
                gl = np.pad(
                    gl, ((0, rows_to_add), (0, 0)), "constant", constant_values=np.nan
                )
            else:
                gl = np.array(gl)
            new_gls_all[i][j].append(gl)

    new_gls_all_array = np.squeeze(np.array(new_gls_all))
    if len(np.shape(new_gls_all_array)) == 3:
        new_gls_all_array = np.expand_dims(new_gls_all_array, axis=0)

    np.save(os.path.join(analysis_dir, "masses.npy"), np.array(masses_all))
    np.save(os.path.join(analysis_dir, 'vafs.npy'), np.array(vafs_all))
    np.save(os.path.join(analysis_dir, "gls.npy"), new_gls_all_array)
    np.save(
        os.path.join(analysis_dir, "midline_x_gls.npy"), np.array(midline_x_gls_all)
    )

    dofs_all = [[h.function_space().mesh().num_vertices() for h in hs] for hs in hs_all]
    np.save(os.path.join(analysis_dir, "dofs.npy"), np.array(dofs_all))


def get_norms_from_outputs(simulation_id):
    ref_fpath = os.path.join(output_dir_path, "uniform_4000", f"outputs-Ice1-id_uniform_4000.h5")
    ref_hs, ref_us = [], []
    with CheckpointFile(ref_fpath, "r") as afile:
        msh = afile.load_mesh("mesh_iter_0_int_0")
        years = np.arange(1, 201, 1)
        for year in years:
            ref_h = afile.load_function(msh, f"thickness_iter_0_int_0_yr_{year:.1f}")
            ref_u = afile.load_function(msh, f"velocity_iter_0_int_0_yr_{year:.1f}")
            ref_hs.append(ref_h)
            ref_us.append(ref_u)
    Q_ref = FunctionSpace(msh, "CG", 1)
    V_ref = VectorFunctionSpace(msh, "CG", 1)
    ref_u_mags = [Function(Q_ref).interpolate(firedrake.sqrt(dot(u, u))) for u in ref_us]

    if os.path.exists(os.path.join(get_analysis_dir(simulation_id), "norms.npy")):
        print(f"Norms already exist for {simulation_id}.")
        return
    simulation_output_fpath = os.path.join(
        output_dir_path, simulation_id, f"outputs-Ice1-id_{simulation_id}.h5"
    )
    print(f"Getting norms for {simulation_id}.")
    analysis_dir = get_analysis_dir(simulation_id)

    rel_diff_h_norms, rel_diff_u_mags_norms = [], []

    num_subintervals = int(simulation_id.split("_")[2])
    years_per_subint = int(200 / num_subintervals)

    # This is very slow
    for i in range(1, 6):
        print(f"--- iteration {i}")
        _rel_diff_h_norms, _rel_diff_u_mags_norms = [], []

        with CheckpointFile(simulation_output_fpath, "r") as afile:
            for j in range(num_subintervals):
                msh = afile.load_mesh(f"mesh_iter_{i}_int_{j}")
                for year in np.arange(j*years_per_subint+1, (j+1)*years_per_subint+1, 1):
                    h = afile.load_function(msh, f"thickness_iter_{i}_int_{j}_yr_{year:.1f}")
                    u = afile.load_function(msh, f"velocity_iter_{i}_int_{j}_yr_{year:.1f}")

                    interp_h = Function(Q_ref).interpolate(h)
                    interp_u = Function(V_ref).interpolate(u)

                    ref_idx = int(year) - 1
                    rel_diff_h = Function(Q_ref).interpolate(abs(interp_h - ref_hs[ref_idx]))
                    rel_diff_u_mag = Function(Q_ref).interpolate(
                        abs(firedrake.sqrt(dot(interp_u, interp_u)) - ref_u_mags[ref_idx])
                    )
                    _rel_diff_h_norms.append(norm(rel_diff_h, "L2") / norm(ref_hs[ref_idx], "L2"))
                    _rel_diff_u_mags_norms.append(norm(rel_diff_u_mag, "L2") / norm(ref_u_mags[ref_idx], "L2"))

        rel_diff_h_norms.append(_rel_diff_h_norms)
        rel_diff_u_mags_norms.append(_rel_diff_u_mags_norms)

        np.save(os.path.join(analysis_dir, "norms.npy"), np.array([rel_diff_h_norms, rel_diff_u_mags_norms]))        


def analyse_all_outputs(output_dir_path, repeat_analysis, get_norms):
    output_dirs = [ 
        f
        for f in os.listdir(output_dir_path)
        if os.path.isdir(os.path.join(output_dir_path, f))
    ]
    print("Analysing simulations:", output_dirs)

    for simulation_id in output_dirs:
        if "800_u-int-h" not in simulation_id:
            continue
        print(f"Analysing {simulation_id}.")
        analysis_exists = analyse_output(simulation_id, repeat_analysis, k=None)
        if get_norms and simulation_id != "uniform_250":
            get_norms_from_outputs(simulation_id)
            continue
        if "uniform" in simulation_id:
            continue
        print(f"Analysis done for {simulation_id}. Plotting results.")
        plot_outputs = plot_output(simulation_id)
        # best_iter_both = summarise_output(simulation_id, *plot_outputs)
        # plot_mesh_evolution(simulation_id, best_iter_both)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir_path>")
        sys.exit(1)
    output_dir_path = os.path.join(os.getcwd(), sys.argv[1])
    repeat_analysis = bool(sys.argv[2]) if len(sys.argv) > 2 else False
    get_norms = bool(sys.argv[3]) if len(sys.argv) > 3 else False
    analyse_all_outputs(output_dir_path, repeat_analysis, get_norms)
