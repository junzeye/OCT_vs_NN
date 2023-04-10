# Make sure to set the correct present working directory
# setwd("path_to_this_subdirectory")
#----------------------------
source("auc_calc.R")
source("dataloader.R")

# How many train test splits
SEEDS = seq(1,100)

# Read from command line the dataset we are interested in
args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
    stop("Invalid input length!")
}
data.v = args[1]

# Load libraries
libraries = c("splitTools","tree","CVXR","pROC","writexl",
              "randomForest", "xpectr", "caret")
suppressMessages( suppressWarnings(sapply(libraries, require, character.only = TRUE)) )

# load standardized data - first column is response variable, col name 'y'
# 1 - cancer, 2 - occupancy, 3 - banknote ; if hinge loss, set isHinge = TRUE
df = load.data(v = data.v)

# Determines stacking trees complexity
inds = partition(df[,1], p = c(train = 0.8, test = 0.2), seed = 0)
train = df[inds$train, ]
test = df[inds$test, ]
parent.tree = tree(y ~., df,
                   control = 
                       tree.control(nrow(df), mincut = 1, 
                                    minsize = 2, mindev = 0)
) # mincut & minsize values set to grow the largest possible tree
max.leaf = count_leaf(parent.tree)
# num.nodes -> uniform stacking tree size (for each train-test partition)
num.nodes = seq(2, max.leaf - 3)

# iterate over many train test splits
auc.sum.rf = 0
auc.sum.tree = 0
auc.sum.stack = 0
success.runs = length(SEEDS)

for (SEED in SEEDS) {
    inds = partition(df[,1], p = c(train = 0.8, test = 0.2), seed = SEED)
    train = df[inds$train, ]
    test = df[inds$test, ]
    folds = create_folds(train[,1], k = 5, m_rep = 3, seed = SEED)
    tryCatch(
        {
            temp = stacking.auc(train, test, folds, inds)
            auc.sum.stack <<- (auc.sum.stack + temp)
        },
        error = function(e){
            success.runs <<- success.runs - 1 
            print(e)
            # record failure (solver timeout), global assignment since inside a function
        }
    )
    suppress_mw( auc.sum.tree <- auc.sum.tree + tree.auc(train, test, folds, inds) )
    suppress_mw( auc.sum.rf <- auc.sum.rf + rf.auc(train, test, folds, inds, SEED) )
    print(sprintf("Seed %d done", SEED))
}

avg.auc.stack = auc.sum.stack / success.runs
avg.auc.rf = auc.sum.rf / length(SEEDS)
avg.auc.tree = auc.sum.tree / length(SEEDS)

sprintf("%d out of %d succeeded (stacking optimization)", success.runs, length(SEEDS))
sprintf("Average test AUC - stack: %.3f", avg.auc.stack)
sprintf("Average test AUC - tree: %.3f", avg.auc.tree)
sprintf("Average test AUC - rf: %.3f", avg.auc.rf)
