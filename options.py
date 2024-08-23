# Description: Options for the ice flow model

def Options(**kwargs):
    options = {
        # half domain
        "Lx": 640e3,
        "Ly": 40e3,
        "dirichlet_ids": tuple([1]),
        "side_wall_ids": tuple([3, 4]),
        "ice_front_ids": tuple([2]),

        # physics constants, in megapascals-meters-years
        "viscosity": 20.0,
        "friction": 1e-2,
        "acc_rate": 0.3,
        "z_0": -100.0,
        "h_c0": 75.0,
        "omega": 0.2,

        # solver parameters
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
            "snes_linesearch_type": "bt",  # cp
        },
        "prognostic_solver_type": "lax-wendroff",
        "prognostic_solver_parameters": {
            "ksp_type": "gmres",
            "pc_type": "sor",
        },

        # petsc_dm_plex_metric_parameters
        "metric_parameters": {
            "dm_plex_metric": {
                "target_complexity": kwargs.get("target_complexity", 800),
                "h_min": 1.0,
                "h_max": 50e3,
                "p": kwargs.get("dm_plex_metric_p", 2.0),
                "a_max": 1e30,
                "hausdorff_number": 1e3,  # doesn't matter since it's a rectangular domain
            }
        } 
    }

    return options
