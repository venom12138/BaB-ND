# BaB-ND: Long-Horizon Motion Planning with Branch-and-Bound and Neural Dynamics

Anonymous ICLR submission. Paper ID: 5364.

## Setup
1. We work on the Ubuntu 22.04.5 LTS operating system and use Miniconda to manage our environment and required packages. Please  install Miniconda if needed: https://docs.conda.io/en/latest/miniconda.html.
2. Create the environment by running the following command. It will create a new environment named `bab-nd` with Python 3.9.17, Torch 2.1.0, and CUDA 12.1. We tested the environment and code on a NVIDIA GeForce RTX 4090.
    ```
    conda env create -f environment.yaml
    ```
3. Activate the environment.
    ```
    conda activate bab-nd
    ```
4. Install the associated package for the Neural Network Verifier $\alpha$-$\beta$ CROWN.
    ```
    cd Verifier_Development
    pip install -e .
    cd ..
    ```

## Instructions

### Synthetic example
1. Run the following command to perform a quick test on the synthetic example:
    ```
    python tasks/func_test/func_test_1.py
    ```
    This will execute 4 methods (GD, MPPI, CEM, and ours) on the synthetic example with an input dimension of 100. Only one random seed is used in the test.
    
    The test generally takes less than 1 minute to complete on an NVIDIA GeForce RTX 4090, and the results will be saved in `output/func_test/quick_test.txt`. The visualization of the function in 1D will be stored in `output/func_test/function_plot_1d.pdf`.
    
    A sample output can be found in `output_sample/func_test`.

2. Run the following command to perform the complete test on the synthetic example:
    ```
    python tasks/func_test/func_test_1.py -complete
    ```
    This will execute 4 methods (GD, MPPI, CEM, and ours) on the synthetic example with input dimensions ranging from 5 to 100, in intervals of 5. Only one random seed is used in the test.

    The test generally takes about 10 minutes to complete on an NVIDIA GeForce RTX 4090, and the results will be saved in `output/func_test/complete_test.pdf`.

    A sample output can be found in `output_sample/func_test`.

### Pushing with Obstacles
Run the following command to try the pushing with obstacles task:

```
bash pushing_w_obs.sh
```

This will run 3 methods (the best 2 baselines, MPPI and CEM, and ours) on 3 cases with different configurations (initial state, target state, and obstacle setting).
- **Note**: To reduce the requirements for computational resources and runtime, we optimize the final step cost considering the obstacles and use a smaller number of samples and iterations for all methods.

The test generally takes about 4 minutes to complete on an NVIDIA GeForce RTX 4090. The results will be stored in `output/pushing_T/`. There will be 3 sub-directories for the 3 cases. Each sub-directory contains the results of the 3 methods. Inside each sub-directory:
- `quantitative.txt` provides the objective of open-loop planning, the final cost after closed-loop execution, and the runtime of open-loop planning for the 3 methods.
- `long_horizon_planning.gif` shows the trajectories of long-horizon open-loop planning from the 3 methods.
- `execution_traj.gif` shows the trajectories of closed-loop execution following the open-loop trajectories.
- (optional) `experiment_results.json` contains detailed information about the experiment.
- **Note**: In the results, MPPI is displayed as "MPPI_BF", CEM is displayed as "DecentCEM" and our method is displayed as "CROWN".

A sample output can be found in `output_sample/pushing_T/`.