SCRIPT := ./mismip.py
ANALYSIS_SCRIPT := ./analyse_outputs.py
FIGURE_SCRIPT := ./figures.py

# set output dir path
OUTPUTS_DIR := ./outputs

get_steady_state:
	python3 $(SCRIPT) \
		--simulation-id steady-state \
		--output-dir $(OUTPUTS_DIR)

RESOLUTIONS = 4000 2000 1000 500 250
uniform_resolution: $(foreach R,$(RESOLUTIONS),uniform_resolution_$(R))

define RUN_UNIFORM_RESOLUTION
uniform_resolution_$(1):
	@echo "Running uniform resolution simulation with resolution $(1)m"
	python3 $(SCRIPT) \
		--simulation-id uniform_$(1) \
		--output-dir $(OUTPUTS_DIR)
endef

$(foreach R,$(RESOLUTIONS),$(eval $(call RUN_UNIFORM_RESOLUTION,$(R))))


C_VALUES = 800 1600 3200 6400 12800
F_VALUES = u h u-int-h tau
adapt_simulation: $(foreach C,$(C_VALUES),$(foreach F,$(F_VALUES),adapt_simulation_$(C)_$(F)))

define RUN_ADAPT_SIMULATION
adapt_simulation_$(1)_$(2):
	@echo "Running simulation with C=$(1) and field=$(2)"
	python3 $(SCRIPT) \
		--simulation-id $(1)_$(2)_20 \
		--output-dir $(OUTPUTS_DIR) \
		--input-steady-state $(OUTPUTS_DIR)/steady-state/outputs-Ice1-id_steady-state.h5 \
		--num-iter 2 \
		--hybrid  # do not pass this for pure global fixed-point approach
endef

# pass the following options for cpu time measurements (flamegraph and no checkpointing)
#		--no-chk
#		-log_view :flamegraph_$(1)_$(2)_20.txt:ascii_flamegraph

$(foreach H,$(C_VALUES),$(foreach U,$(F_VALUES),$(eval $(call RUN_ADAPT_SIMULATION,$(H),$(U)))))

analyse:
	python3 $(ANALYSIS_SCRIPT) $(OUTPUTS_DIR) 0 1

# requirement for figures: https://github.com/callumrollo/cmcrameri
FIGURES = metric_components schemes initial_steady_state uniform_convergence \
          strat_comparison strat_comparison_meshes hessian_meshes \
          hessian_aspect_ratio hessian_err hessian_cpu_time

make_figures:
	@for fig in $(FIGURES); do \
		python3 $(FIGURE_SCRIPT) --output-dir $(OUTPUTS_DIR) --fig $$fig; \
	done
