# ORFE Senior Thesis Code

### Introduction

This is the codebase for my senior thesis, a report on computational experiments I conducted on several tree-based classification algorithms. For this thesis project, I was forunate to have been mentored by Professor [Jason Klusowski](https://jasonklusowski.github.io/) at Princeton ORFE. Each folder in this repo contains scripts I used to generate results presented in the corresponding thesis chapter. 

If you'd like to rerun some experiments, make sure to first set your present working directory to the root of this repository (`Senior_Thesis_Code/`). When I reorganized my code scripts by chapters, I tried my best to refactor module and file references assuming you are at the repo root. The only exception is the experiments for Chapter 5, which needs you to set the working directory to `/Senior_Thesis_Code/Ch5_stacking`.

### Repository Contents

- `Ch2_framework/`: Builds a pretrained feedforward neural network of a customizable dimension, and uses the neural net to generate synthetic data. Stores the synthetic data in the folder `synthetic_data`.
    - `make_width_data.ipynb` is the driver code.
- `Ch3_optimal/`: Trains and tests optimal classification trees (OCTs) on the previously mentioned synthetic data.
    - `experiment_instance.py` is the main script.
    - `slurm_scripts/` contains all the Slurm scripts that intialize the parameters of the optimization instance specified in `experiment_instance.py` and submits the computing job to Princeton's computing cluster to solve the optimization problem.
    - `tree/`: module that implements the OCT optimization algorithm. Forked from [this repo](https://github.com/LucasBoTang/Optimal_Classification_Trees) with minor revisions.
- `Ch4_oblique/`: Implements an oblique classification tree algorithm (HHCART). Also trains and tests oblique classification trees on the previously mentioned synthetic data.
    - `scikit_obliquetree/`: Implements the HHCART algorithm for classification tasks, which builds an oblique decision tree classifier. Forked from [this repo](https://github.com/hengzhe-zhang/scikit-obliquetree) with several adaptations to generalize the module for classification tasks.
    - `oblique.ipynb` is the driver code. If it takes too long to run on the Jupyter notebook, try using `oblique_experiment.py` instead.
    - `VIS_Oct6.ipynb`: Responsible for generating Figures 4.2 and 4.3 in the senior thesis.
- `Ch5_stacking/`: Implements stacking tree predictors for binary classification. Compares test performances of stacking tree, single tree and random forests on three different datasets.
    - `main.R` is the driver code. Use the command `Rscript main.R number` where `number` = 1, 2, or 3 depending on the dataset you want to analyze. The default loss for stacking is logistic.
    - `auc_calc.R` provides functions that calculate the test AUC scores for stacking, single CART tree, and random forests, respectively. The AUC scoring function for stacking also implements the stacking algorithm as a convex optimization program.
    - `nolambda_breastCancer_hinge.Rmd`: implements stacking tree with __hinge__ loss. Responsible for generating Figure 5.2 in the senior thesis.
    - `nolambda_breastCancer_logistic.Rmd`: implements stacking tree with __logistic__ loss. Responsible for generating Figure 5.1 in the senior thesis.
- `Ch6_visual/`: Runs visualization experiments that compare the decision boundaries of deep neural networks with those of oblique and axis-parallel decision trees.
    - `scratch.ipynb`: driver code for experiments done on `sklearn`'s half moon dataset. Corresponds to thesis chapter 6.1-6.3.
    - `scratch_circle.ipynb`: driver code for experiments done on `sklearn`'s concentric circles dataset. Corresponds to thesis chapter 6.4.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [R 4.1.1](https://cran.r-project.org/bin/windows/base/old/4.1.1/)
* [Gurobi 9.1](https://www.gurobi.com/)
* [scikit-learn 0.24.2](https://scikit-learn.org/)
* [SciPy 1.6.3](https://www.scipy.org/)

### Code Credits

I want to sincerely thank those from the open source community for allowing me to utilize their modules and code, saving me from reinventing the wheels. Below, I list code scripts and modules in this repository where a major portion comes from a public codebase, as well as its authors' information:

- `Ch2_framework/dataset.py` and `Ch3_optimal/tree/`. Source: Bo Tang & Bo Lin. "Optimal_Classification_Trees". GitHub (2021). [Link](https://github.com/LucasBoTang/Optimal_Classification_Trees).
- `Ch4_oblique/scikit_obliquetree/`. Source: Hengzhe Zhang, "scikit-obliquetree". GitHub (2021). [Link](https://github.com/hengzhe-zhang/scikit-obliquetree).
- Python and R libraries, such as `sklearn`, `numpy`, `CVXR`, etc. Optimization solvers like `Gurobi`, `ECOS`, etc.

This is by no means an exhaustive list. I do my best to give credits and apologize for any accidental omission. Please feel free to let me know if I forgot to acknowledge your work.

### Algorithm Credits

- D. Bertsimas & J. Dunn. Optimal classification trees. Machine Learning, 106(7), 1039-1082 (2017).
- L. Breiman. Stacked regressions. Mach Learn 24, 49-64 (1996).
- D.C. Wickramarachchi, B.L. Robertson, M. Reale, C.J. Price, J. Brown. HHCART: An oblique decision tree,
Computational Statistics & Data Analysis, Volume 96, 12-23 (2016).

### Data Credits

- William Wolberg, Nick Street, and Olvi Mangasarian. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repo (1995). [Link](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- Volker Lohweg & Helene Darkse. Banknote Authentication Data Set. UCI Machine Learning Repo (2013). [Link](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- Luis Candanedo. Occupancy Detection Data Set. UCI Machine Learning Repo (2016). [Link](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)