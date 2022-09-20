# Optimal Classification Tree
 
### Introduction

We plan to investigate the modeling power of optimally grown classification trees in comparison with feedforward neural networks. We will first obtain an empirical relationship between the model complexity of a neural network versus that of a classification tree when both models have similar test accuracy. We will also look for hybrid tree algorithms that strike a balance between optimality and runtime, in part to help bound the performance of optimally grown classification trees.

### Dependencies

* [Python 3.7](https://www.python.org/)
* [Gurobi 9.1](https://www.gurobi.com/)
* [scikit-learn 0.24.2](https://scikit-learn.org/)
* [SciPy 1.6.3](https://www.scipy.org/)

### MIP Models

- [Optimal Classification Trees (OCT)](https://link.springer.com/article/10.1007/s10994-017-5633-9) - Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. Machine Learning, 106(7), 1039-1082.

### Parameters

- **max_depth:** The maximum depth of the tree.
- **min_samples_split:** The minimum number of samples required to split an internal node.
- **alpha:** The penalty term coefficients for the number of splitting nodes.
- **warmstart**: Warm start with skLearn decision tree or not.
- **timelimit**: The time limit for running the MIP solver.
- **output**: Show the optimizing output or not.

### Sample Code

TBD

### Data

TBD

### Credits

I would like to thank Lucas Tang for making his implementation of the OCT algorithm open-source. His codebase forms the backbone of this project, and you can look at his code [here](https://github.com/LucasBoTang/MIP_Decision_Tree/blob/main/Report.pdf).