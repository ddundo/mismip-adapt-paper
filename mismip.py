import argparse
import os
import time

from animate.adapt import adapt
from firedrake import *
# from glac_adapt.figures.pingpong_v4_u import V_A
from goalie_adjoint import *

# get variational forms from icepack
from icepack.models import IceStream
from icepack.solvers import FlowSolver

# for adaptors
import utility_functions as uf
from options import Options

# Citations.print_at_exit()

parser = argparse.ArgumentParser()
parser.add_argument("--simulation-id", type=str, required=True, help="Simulation ID in the format '<complexity>_<field>_<numSubintervals>', 'uniform_<resolution in metres>', or 'steady-state'.")
parser.add_argument("--output-dir", type=str, help="Output directory.")
parser.add_argument("--input-steady-state", type=str, help="Path to steady state checkpoint file.")
parser.add_argument("--num-iter", type=int, default=1, help="Number of iterations.")
parser.add_argument("--num-exports-per-year", type=int, default=1, help="Number of solution exports per year.")
parser.add_argument("--no-chk", action="store_true", help="Do not export solutions.")
parser.add_argument("--hybrid", action="store_true", help="Use hybrid mesh adaptation algorithm.")

# args = parser.parse_args()
args, _ = parser.parse_known_args()

simulation_id = args.simulation_id
id_components = simulation_id.split("_")
steady_state_simulation = simulation_id == "steady-state"
uniform_simulation = "uniform" in simulation_id
if steady_state_simulation:
    num_subintervals = 5
    target_complexity = None
    adapt_field = None
elif uniform_simulation:
    resolution = int(id_components[1])
    level = int(np.log2(4000/resolution))  # number of uniform refinements
    num_subintervals = 1
    target_complexity = None
    adapt_field = None
else:
    simulation_id = simulation_id + f"_{'hybrid' if args.hybrid else 'global'}"
    level = 0
    target_complexity = int(id_components[0])
    adapt_field = id_components[1]
    assert adapt_field in ["h", "u", "u-int-h", "tau"], "Invalid adapt field."
    num_subintervals = int(id_components[2])

num_iter = args.num_iter

output_dir = os.path.join(args.output_dir, simulation_id)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

options_kwargs = {
    "target_complexity": target_complexity,
}
options = Options(**options_kwargs)

# # copy files to the backup directory
# backup_dir = os.path.join(output_dir, "backup")
# os.makedirs(backup_dir)
# backup_files = [
#     "ice1.py",
#     "options.py",
#     "utility_functions.py",
# ]
# for f in backup_files:
#     fpath = os.path.join(os.path.dirname(__file__), f)
#     os.system(f"cp {fpath} {backup_dir}")
# print("Copied files to backup directory.")

output_fpath = os.path.join(output_dir, f"outputs-Ice1-id_{simulation_id}.h5")
if os.path.exists(output_fpath):
    raise ValueError(f"Output file {output_fpath} already exists.")

if steady_state_simulation:
    meshes = [
        RectangleMesh(
            int(options["Lx"] / options["Ly"] * 10 * 2**i),
            int(10 * 2**i),
            options["Lx"],
            options["Ly"],
            name=f"mesh_iter_0_int_{i}",
        )
        for i in range(num_subintervals)
    ]

    end_time = 15e3
    timesteps = [2, 0.5, 1/24, 1/24, 1/24]
    num_timesteps_per_export = 1/(np.array(timesteps)*args.num_exports_per_year/1e3)
    num_timesteps_per_export = [int(n) for n in num_timesteps_per_export]
    subintervals = [(0, 5e3), (5e3, 9e3), (9e3, 12e3), (12e3, 14e3), (14e3, 15e3)]

    x = SpatialCoordinate(meshes[0])[0]
    Q_fspace = FunctionSpace(meshes[0], "CG", 1)
    V_fspace = VectorFunctionSpace(meshes[0], "CG", 1)
    R_fspace = FunctionSpace(meshes[0], "R", 0)
    input_h = Function(Q_fspace).interpolate(Function(R_fspace).assign(100.0))
    input_u = Function(V_fspace).interpolate(as_vector((90 * x / options["Lx"], 0)))
else:
    meshes = [
        RectangleMesh(
            int(options["Lx"] / options["Ly"] * 10 * 2**level),
            int(10 * 2**level),
            options["Lx"],
            options["Ly"],
            name=f"mesh_iter_0_int_{i}",
        )
        for i in range(num_subintervals)
    ]

    end_time = 200.0
    timesteps = 1/24
    num_timesteps_per_export = int(1/(timesteps*args.num_exports_per_year))
    subintervals = None

    with CheckpointFile(args.input_steady_state, "r") as afile:
        input_mesh = afile.load_mesh("mesh_iter_4_int_0")
        input_h = afile.load_function(input_mesh, name="thickness_steady")
        input_u = afile.load_function(input_mesh, name="velocity_steady")

subinterval_length = end_time / num_subintervals
time_partition = TimePartition(
        end_time,
        num_subintervals,
        timesteps,
        ["u", "h"],
        subintervals=subintervals,
        field_types=["steady", "unsteady"],
        num_timesteps_per_export=num_timesteps_per_export,
    )

params_args = {
    "element_rtol": 0.005,
    "maxiter": num_iter,
    "miniter": num_iter,
}
parameters = MetricParameters(params_args)

print("Output file:", output_fpath)
print("Initial num_cells:", meshes[0].num_cells())

##############################################################

icepack_model = IceStream(friction=uf.friction_law)
icepack_solver = FlowSolver(icepack_model, **options)
prognostic_solver = icepack_solver._prognostic_solver
diagnostic_solver = icepack_solver._diagnostic_solver


def get_function_spaces(mesh):
    fspaces = {
        "u": VectorFunctionSpace(mesh, "CG", 1),
        "h": FunctionSpace(mesh, "CG", 1),
    }
    return fspaces

def get_form(mesh_seq):
    def form(index, **auxiliary_fields):
        u, u_ = mesh_seq.fields["u"]
        h, h_ = mesh_seq.fields["h"]

        R = FunctionSpace(mesh_seq[index], "R", 0)

        if "err_ind_time" in auxiliary_fields or auxiliary_fields=={}:
            try:
                sim_time = auxiliary_fields["err_ind_time"]
            except Exception:
                sim_time = 0.0

            Q = mesh_seq.function_spaces["h"][index]
            z_b = assemble(interpolate(uf.mismip_bed_topography(mesh_seq.meshes[index]), Q))
            h0 = Function(Q).assign(h_)
            h0.rename("thickness_inflow")  # Rename to avoid confusion in taping
            s = Function(Q).interpolate(uf.surface_expression(h_, z_b))

            acc_rate = Function(R).assign(options["acc_rate"])
            if sim_time < 100.0:
                omega = Function(R).assign(options["omega"])
                z_0 = Function(R).assign(options["z_0"])
                h_c0 = Function(R).assign(options["h_c0"])
                z_d = s - h_
                h_c = z_d - z_b
                melt = omega * tanh(h_c / h_c0) * max_value(z_0 - z_d, 0.0)
                a = Function(Q).interpolate(acc_rate - melt)
            else:
                a = Function(Q).interpolate(acc_rate)

        else:
            s = auxiliary_fields["surface"]
            a = auxiliary_fields["accumulation"]
            h0 = auxiliary_fields["thickness_inflow"]

        solver_fields = {
            "velocity":u,
            "thickness":h,
            "thickness_old":h_,
            "surface":s,
            "accumulation":a,
            "thickness_inflow":h0,
            "timestep":Function(R).assign(mesh_seq.time_partition.timesteps[index]),
            "friction":Function(R).assign(options["friction"]),
            "fluidity":Function(R).assign(options["viscosity"]),
        }

        prognostic_solver.setup(**solver_fields)
        F_h = prognostic_solver.F

        diagnostic_solver.setup(**solver_fields)
        F_u = diagnostic_solver.F

        return {"u": F_u, "h": F_h}
    return form

def get_solver(mesh_seq):
    def solver(index):
        tp = mesh_seq.time_partition
        time, _ = tp.subintervals[index]
        dt = tp.timesteps[index]
        num_steps = tp.num_timesteps_per_subinterval[index]

        V = mesh_seq.function_spaces["u"][index]
        Q = mesh_seq.function_spaces["h"][index]

        u, u_ = mesh_seq.fields["u"]
        h, h_ = mesh_seq.fields["h"]
        u.assign(u_)
        h.assign(h_)
        h0 = Function(Q).assign(h_)

        if steady_state_simulation:
            print(f'start of subinterval {index}:', uf.get_mass(h))

        z_b = Function(Q).interpolate(uf.mismip_bed_topography(mesh_seq[index]))
        s = Function(Q).interpolate(uf.surface_expression(h, z_b))
        a = Function(Q)

        R = FunctionSpace(mesh_seq[index], "R", 0)
        z_0 = Function(R).assign(options["z_0"])  # cut-off base elevation
        h_c0 = Function(R).assign(options["h_c0"])  # cut-off water column thickness
        acc_rate = Function(R).assign(options["acc_rate"])
        omega = Function(R).assign(options["omega"])

        melting = float(time) < 100.0 and not steady_state_simulation
        if not melting:
            a.interpolate(acc_rate)

        auxiliary_fields = {
                "surface": s,
                "accumulation": a,
                "thickness_inflow": h0,
            }
        forms = mesh_seq.form(index, **auxiliary_fields)
        F_h = forms["h"]
        nlvp_h = NonlinearVariationalProblem(F_h, h)
        nlvs_h = NonlinearVariationalSolver(
            nlvp_h,
            solver_parameters=options["prognostic_solver_parameters"],
            ad_block_tag="h",
        )

        F_u = forms["u"]
        bcs = DirichletBC(V, (1e-8, 1e-8), options["dirichlet_ids"])
        degree_u = u.ufl_element().degree()
        degree_h = h.ufl_element().degree()
        quad_degree = 3*(degree_u - 1) + 2 * degree_h
        nlvp_u = NonlinearVariationalProblem(
            F_u, u, bcs,
            form_compiler_parameters={"quadrature_degree": quad_degree}
        )
        nlvs_u = NonlinearVariationalSolver(
            nlvp_u,
            solver_parameters=options["diagnostic_solver_parameters"],
            ad_block_tag="u",
        )

        for _ in range(num_steps):
            if melting and float(time) > 100.0:
                z_d = s - h  # elevation of the ice base
                h_c = z_d - z_b  # water column thickness
                melt = omega * tanh(h_c / h_c0) * max_value(z_0 - z_d, 0.0)
                a.interpolate(acc_rate - melt)

            nlvs_u.solve()
            nlvs_h.solve()
            if h.dat.data.min() < 0.0:
                print(f"Negative thickness at {float(time)} a: {h.dat.data.min()}")
                h.dat.data[:] = np.maximum(h.dat.data, 0.0)
            s.interpolate(uf.surface_expression(h, z_b))

            h_.assign(h)

            time += dt

        if steady_state_simulation:
            print(f'end of subinterval {index}:', uf.get_mass(h))

        return {"u": u, "h": h}
    return solver

def get_initial_condition(mesh_seq):
    h = mesh_seq._transfer(input_h, mesh_seq.function_spaces["h"][0])
    u = mesh_seq._transfer(input_u, mesh_seq.function_spaces["u"][0])

    return {"u": u, "h": h}

mp = {key: value for key, value in options["metric_parameters"].items()}
subinterval_metric_fns = {
    "h": uf.h_metric,
    "u": uf.u_metric,
    "tau": uf.tau_metric,
    "u-int-h": uf.uh_metric,
}
def get_metric_fn(adapt_field): return subinterval_metric_fns[adapt_field]

@PETSc.Log.EventDecorator()
def global_adaptor(mesh_seq, solutions):
    mesh_seq.converged[:] = False
    metrics = []

    for i in range(len(mesh_seq)):
        hs = solutions["h"]["forward"][i]
        us = solutions["u"]["forward"][i]
        subinterval_metrics = []

        for j in range(len(hs)):
            h = hs[j]
            u = us[j]

            metric_fn = get_metric_fn(adapt_field)
            _metric = metric_fn(h, u, mp)

            _metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
            subinterval_metrics.append(_metric)

        combined_metric = subinterval_metrics[0].copy(deepcopy=True)
        for m in subinterval_metrics[1:]:
            combined_metric.intersect(m)

        metrics.append(combined_metric)

    space_time_normalise(metrics, mesh_seq.time_partition, mp)

    prev_iter = int(mesh_seq[0].name.split("mesh_iter_")[1].split("_")[0])
    for i, metric in enumerate(metrics):
        mesh_seq[i] = adapt(
            mesh_seq[i], metric, name=f"mesh_iter_{prev_iter+1}_int_{i}"
        )

    if not args.no_chk:
        print(f"Checkpointing adapted meshes {prev_iter+1} and solutions {prev_iter}.")
        uf.checkpoint_meshseq(mesh_seq, save_sols=True,
                            save_inds=False,
                            iteration=prev_iter, metrics=metrics)

    return True

@PETSc.Log.EventDecorator()
def global_adaptor_utau(mesh_seq, solutions):
    mesh_seq.converged[:] = False
    metrics_tau, metrics_u = [], []

    for ii, f in enumerate(["tau", "u"]):
        metrics = metrics_tau if f == "tau" else metrics_u
        for i in range(len(mesh_seq)):
            h_sols = solutions["h"]["forward"][i]
            u_sols = solutions["u"]["forward"][i]
            subinterval_metrics = []

            for j in range(len(h_sols)):
                h = h_sols[j]
                u = u_sols[j]

                metric_fn = get_metric_fn(f)
                _metric = metric_fn(h, u, mp)

                _metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
                subinterval_metrics.append(_metric)

            combined_metric = subinterval_metrics[0].copy(deepcopy=True)
            for m in subinterval_metrics[1:]:
                combined_metric.intersect(m)

            metrics.append(combined_metric)

        space_time_normalise(metrics, mesh_seq.time_partition, mp)

    metrics = []
    for m1, m2 in zip(metrics_tau, metrics_u):
        intersected_m = m1.copy(deepcopy=True).intersect(m2)
        metrics.append(intersected_m)

    prev_iter = int(mesh_seq[0].name.split("mesh_iter_")[1].split("_")[0])
    for i, metric in enumerate(metrics):
        mesh_seq[i] = adapt(
            mesh_seq[i], metric, name=f"mesh_iter_{prev_iter+1}_int_{i}"
        )

    if not args.no_chk:
        print(f"Checkpointing adapted meshes {prev_iter+1} and solutions {prev_iter}.")
        uf.checkpoint_meshseq(mesh_seq, save_sols=True,
                            save_inds=False,
                            iteration=prev_iter, metrics=metrics)

    return True

@PETSc.Log.EventDecorator()
def classical_adaptor(mesh_seq, i):
    Q = mesh_seq.function_spaces["h"][i]
    V = mesh_seq.function_spaces["u"][i]

    if i == 0:
        ic = mesh_seq.initial_condition
        h = ic["h"]
        u = ic["u"]
    else:
        solutions = mesh_seq.solutions.extract()
        h = mesh_seq._transfer(solutions["h"]["forward"][i-1][-1], Q)
        u = mesh_seq._transfer(solutions["u"]["forward"][i-1][-1], V)

    metric_fn = get_metric_fn(adapt_field)
    metric = metric_fn(h, u, mp)
    metric.normalise()

    prev_iter = int(mesh_seq[i].name.split("mesh_iter_")[1].split("_")[0])
    mesh_seq[i] = adapt(mesh_seq[i], metric, name=f"mesh_iter_{prev_iter+1}_int_{i}")

@PETSc.Log.EventDecorator()
def classical_adaptor_utau(mesh_seq, i):
    Q = mesh_seq.function_spaces["h"][i]
    V = mesh_seq.function_spaces["u"][i]

    if i == 0:
        ic = mesh_seq.initial_condition
        h = ic["h"]
        u = ic["u"]
    else:
        solutions = mesh_seq.solutions.extract()
        h = mesh_seq._transfer(solutions["h"]["forward"][i-1][-1], Q)
        u = mesh_seq._transfer(solutions["u"]["forward"][i-1][-1], V)

    u_metric = get_metric_fn("u")(h, u, mp)
    tau_metric = get_metric_fn("tau")(h, u, mp)
    u_metric.normalise()
    tau_metric.normalise()
    metric = u_metric.copy(deepcopy=True).intersect(tau_metric)

    prev_iter = int(mesh_seq[i].name.split("mesh_iter_")[1].split("_")[0])
    mesh_seq[i] = adapt(mesh_seq[i], metric, name=f"mesh_iter_{prev_iter+1}_int_{i}")


mesh_seq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    parameters=parameters,
    transfer_method="project",
    transfer_kwargs={"bounded": not steady_state_simulation},
)
mesh_seq.output_fpath = output_fpath

##############################################################

if steady_state_simulation or uniform_simulation:
    t0 = time.time()
    mesh_seq.solve_forward()
    t1 = time.time()
    print(f"Total time taken for {simulation_id}: {(t1-t0)/60} min")
    with open(os.path.join(args.output_dir, "time.txt"), "a") as f:
        f.write(f"{simulation_id},{(t1-t0)/60}\n")
    if steady_state_simulation:
        with CheckpointFile(output_fpath, "w") as chk:
            chk.save_mesh(mesh_seq.meshes[-1])
            chk.save_function(mesh_seq.fields["u"][0], name="velocity_steady")
            chk.save_function(mesh_seq.fields["h"][0], name="thickness_steady")
    elif not args.no_chk:
        mesh_seq.output_fpath = output_fpath
        uf.checkpoint_meshseq(mesh_seq, True)
    exit()

classical_adaptor = classical_adaptor_utau if adapt_field == "u-int-tau" else classical_adaptor
global_adaptor = global_adaptor_utau if adapt_field == "u-int-tau" else global_adaptor

t0 = time.time()
if args.hybrid:
    # one iteration of the classical mesh adaptation algorithm
    solutions = mesh_seq.adapt_on_the_fly(classical_adaptor)
    if not args.no_chk:
        uf.checkpoint_meshseq(mesh_seq, save_sols=True)

    # adapt the entire mesh sequence at the end of the first iteration
    global_adaptor(mesh_seq, solutions)
    if not args.no_chk:
        uf.checkpoint_meshseq(mesh_seq, save_sols=False)
else:
    print("Skipping classical adaptation.")

# Global fixed point adaptation algorithm
mesh_seq.fixed_point_iteration(global_adaptor)

# Solve on the final mesh sequence
mesh_seq.solve_forward()
if not args.no_chk:
    uf.checkpoint_meshseq(mesh_seq, True)
t1 = time.time()
print(f"Total time taken for {simulation_id}: {(t1-t0)/60} min")

with open(os.path.join(args.output_dir, "time.txt"), "a") as f:
    f.write(f"{simulation_id},{(t1-t0)/60}\n")

print(f"Finished simulation {simulation_id}.")
