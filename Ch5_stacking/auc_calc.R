# Helper function - calculates number of leaves in a tree
count_leaf <- function(t) {
    isLeaf = t$frame$var == "<leaf>" # TRUE if node is leaf
    return(sum(isLeaf))
}

# function for calculating AUC scores for stacking
# depends on num.nodes (a global variable)
stacking.auc <- function(f.train, f.test , f.folds, f.inds){
    
    preds = matrix(0, nrow(f.train), length(num.nodes))
    # stratified K-fold CV
    for (fold in f.folds){
        valid.set = f.train[-fold, ]
        nrow.valid = nrow(valid.set)
        # genesis tree to be pruned
        parent.tree = tree(y ~., f.train[fold, ],
                           control = 
                               tree.control(nrow(f.train[fold, ]), mincut = 1, 
                                            minsize = 2, mindev = 0)
        )
        for (i in seq_along(num.nodes)) {
            curr.tree = prune.tree(parent.tree, best = num.nodes[i])
            preds[-fold, i] = predict(curr.tree, valid.set)
        }
    }
    
    alphas = Variable(length(num.nodes)) # CVXR variables
    preds.weighted = preds %*% alphas
    y = f.train[, 1]
    
    # set up CVXR optimization instance
    obj = (1 / nrow(f.train)) * sum_entries( -y * log(preds.weighted) - (1-y) * log(1-preds.weighted) )
    constraint = list(alphas >= 0)
    prob = Problem(Minimize(obj), constraint)
    solution = solve(prob, solver = "ECOS", num_iter = 1000)
    
    preds = matrix(0, nrow(df), length(num.nodes))
    # genesis tree to be pruned
    parent.tree = tree(y ~., f.train,
                       control = 
                           tree.control(nrow(f.train), mincut = 2, 
                                        minsize = 5, mindev = 0.0001)
    )
    
    for (i in seq_along(num.nodes)) {
        curr.tree = prune.tree(parent.tree, best = num.nodes[i])
        preds[, i] = predict(curr.tree, df)
    }
    
    alphas.opt = solution$getValue(alphas) # optimal value of alphas
    y = f.train[, 1]
    
    test.preds = preds[f.inds$test, ] %*% alphas.opt
    suppress_mw( test.auc <- auc(f.test$y ~ c(test.preds)) )
    return(test.auc)
}

# function for calculating AUC scores for trees
tree.auc <- function(f.train, f.test , f.folds, f.inds){
    cv.maxnode = num.nodes
    cv.AUC = numeric( length(cv.maxnode) )
    
    for (j in seq_along(cv.maxnode) ) {
        maxnode = cv.maxnode[j]
        auc.score = 0
        
        # stratified K-fold CV
        for (fold in f.folds){
            valid.set = f.train[-fold, ]
            nrow.valid = nrow(valid.set)
            # genesis tree to be pruned
            parent.tree = tree(y ~., f.train[fold, ],
                               control = 
                                   tree.control(nrow(f.train[fold, ]), mincut = 1, 
                                                minsize = 2, mindev = 0)
            )
            pruned.tree = prune.tree(parent.tree, best = maxnode)
            pruned.preds = predict(pruned.tree, valid.set)
            suppress_mw( auc.score <- auc.score + auc(valid.set$y, pruned.preds) )
        }
        cv.AUC[j] = auc.score / length(f.folds) # average AUC value
    }
    
    # Retrain with optimal terminal node count and evaluate test performance
    best.maxnode = cv.maxnode[which.max(cv.AUC)]
    parent.tree = tree(y ~., f.train,
                       control = 
                           tree.control(nrow(f.train), mincut = 1, 
                                        minsize = 2, mindev = 0)
    )
    pruned.tree = prune.tree(parent.tree, best = best.maxnode)
    pruned.preds = predict(pruned.tree, f.test)
    return(auc(f.test$y, pruned.preds))
}

# function for calculating AUC scores for random forests
rf.auc <- function(f.train, f.test , f.folds, f.inds, s){
    set.seed(s)
    control <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")
    tunegrid <- expand.grid(.mtry=seq(1, ncol(f.train) - 1))
    rf_gridsearch <- train(f.train[,-1], factor(f.train[,1]), ntree=256, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control)
    best_mtry = rf_gridsearch$bestTune[[1]]
    rf.best = randomForest(y ~., data = f.train, mtry = best_mtry, ntree = 256)
    rf.test.preds = predict(rf.best, f.test)
    return(auc(f.test$y, rf.test.preds))
}