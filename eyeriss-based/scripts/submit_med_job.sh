# Environment variable
#    SCRATCH_BASE: point to the base scratch directory that stores the jobs_out and jobs_err directories

EXP_SCRIPT_NAME=$1

qsub --projectMode direct -P research_arch_misc \
         -e $SCRATCH_BASE/jobs_err/job%J.err \
         -o $SCRATCH_BASE/jobs_out/job%J.out \
         -n 8 \
         -q o_cpu_8G_8H \
         sh ${EXP_SCRIPT_NAME}
