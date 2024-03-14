#!/bin/bash

pip install -r requirements.txt

#########################################################################
#                           EVALUATION RUNS                             #
#########################################################################
# Section 5.1
python eval.py rl --nenvs 30 --nruns 1 --congested f --demand_factor 1

# Section 5.2
python eval.py ts --nenvs 30 --nruns 10 --congested f --demand_factor 1
python eval.py ga --nenvs 30 --nruns 10 --congested f --demand_factor 1

# Appendix A.1
python eval.py ts --nenvs 30 --nruns 10 --congested t --demand_factor 1

# Appendix A.2
python eval.py ts --nenvs 30 --nruns 1 --congested t --demand_factor 0.5
python eval.py ts --nenvs 30 --nruns 1 --congested t --demand_factor 1
python eval.py ts --nenvs 30 --nruns 1 --congested t --demand_factor 2


#########################################################################
#                           GENERATE FIGURES                            #
#########################################################################

# Figure 4
python gen_f4_maps.py

# Figure 5
python plot_eval_output.py results/results_*_E30_R10_DF1.0_NC* \
    --sort_instance_by_objective rl \
    --legends GA RL TS \
    --output_prefix figures/fig_5

# Figure 6
python gen_f6_compare_worse_run.py

# Figure A.7
python plot_eval_output.py results/results_ts_E30_R10_DF0.01* \
    --sort_instances_by_objective False \
    --grouped_by congested \
    --legends LSSDP-C LSSDP-NC \
    --output_prefix figures/fig_a7

# Figure A.8
python gen_fa8_objective_initial_load_factor.py

#########################################################################
#                            GENERATE TABLES                            #
#########################################################################

# Table 4
python gen_t4_mean_objectives.py

# Table A.5
python gen_ta5_compare_cnc.py