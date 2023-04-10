# ORFE Senior Thesis Code

### Introduction

This is the codebase for my senior thesis, a report on computational experiments I performed on several tree-based classification algorithms. Each folder contains code I used to generate results presented in the corresponding thesis chapter. 

If you'd like to replicate the results, please take the time to map out the interdepencies between the code scripts. Certain references to modules/files might have become outdated after I reorganized my code scripts by chapters. For example, to run the experiments in the folders `Ch3_optimal` and `Ch4_oblique`, you will need to access the subfolder `synthetic_data` inside the folder `Ch2_framework`, which might require you to change some pathnames in my scripts.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [R 4.1.1](https://cran.r-project.org/bin/windows/base/old/4.1.1/)
* [Gurobi 9.1](https://www.gurobi.com/)
* [scikit-learn 0.24.2](https://scikit-learn.org/)
* [SciPy 1.6.3](https://www.scipy.org/)

### Code Credits

I want to sincerely thank those from the open source community for allowing me to utilize their modules and code, saving me from reinventing the wheels. Below, I list the code scripts and modules in this repository where a major portion comes from a public codebase, as well as the authors' information:

- `Ch2_framework/dataset.py`, `Ch3_optimal/tree/`, `Ch3_optimal/Experiments.ipynb`, and `res/`. Source: Bo Tang & Bo Lin. "Optimal_Classification_Trees". GitHub (2021). [Link](https://github.com/LucasBoTang/Optimal_Classification_Trees).
- `Ch4_oblique/scikit_obliquetree/`. Source: Hengzhe Zhang, "scikit-obliquetree". GitHub (2021). [Link](https://github.com/hengzhe-zhang/scikit-obliquetree).
- Python and R libraries, such as `sklearn`, `numpy`, `CVXR`, etc. Optimization solvers like `Gurobi`, `ECOS`, etc.

This is by no means an exhaustive list. I do my best to give credits and apologize for any accidental omission. Please feel free to let me know if I forgot to acknowledge your work.

### Algorithm Credits

- D. Bertsimas & J. Dunn. Optimal classification trees. Machine Learning, 106(7), 1039-1082 (2017).
- L. Breiman. Stacked regressions. Mach Learn 24, 49¨C64 (1996).
- D.C. Wickramarachchi, B.L. Robertson, M. Reale, C.J. Price, J. Brown. HHCART: An oblique decision tree,
Computational Statistics & Data Analysis, Volume 96, 12-23 (2016).

### Data Credits

- William Wolberg, Nick Street, and Olvi Mangasarian. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repo (1995). [Link](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- Volker Lohweg & Helene Darkse. Banknote Authentication Data Set. UCI Machine Learning Repo (2013). [Link](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- Luis Candanedo. Occupancy Detection Data Set. UCI Machine Learning Repo (2016). [Link](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)