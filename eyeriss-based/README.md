Eyeriss-based experimental setup
------------------------------------------------
This folder contains the necessary files for performing eyeriss-based architecture
design space exploration by mix-and-matching various sparse optimizations at different
architecture levels.

### File Structure
This directory contains the following subdirectories:
1. `arch`: stores the template architecture specification of eyeriss-168 architecture
2. `components`: accelergy compound components needed to fully define the architecture
3. `variables`: hardware attributes for each component in the architecture specification.
Each architecture variation has its own set of variables specification.
4. `sparse_opt`: sparse optimization specifications for each architecture level. Each architecture
variation has its own set of sparse optimization specification. Sparse optimization file and
variables files with the same name are designed for the same architecture variation.
5. `constraints`: map space constraints for eyeriss row stationary dataflow.
6. `mapper`: optimization criteria, search algorithm, etc.
7. `layer_shapes`: layer shape specification for AlexNet Sparse model.
8. `scripts`: scripts for running the sweeping of architecture variations.

### Run Experiments
To sweep the architectures with different set of optimizations applied (each architecture is evaluated
on the 5 conv layers of AlexNet):

```
cd scripts
./submit_med_job.sh mix_match.sh
```
Outputs of the sweep should be saved to `$SIM/eyeriss`, each architecture variation's results 
are saved in the subfolder with its architecture name

