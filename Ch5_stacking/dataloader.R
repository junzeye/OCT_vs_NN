# The value of y (response variable) is set to be {0,1} -> default for logistic loss
load.data <- function(v, isHinge = FALSE){
    
    if (!(v %in% c(1,2,3)) ) {
        stop("invalid dataset number") # throw exception
    }
    
    # Wisconsin breast cancer data
    if (v == 1) {
        df = read.csv("wdbc.data", header = FALSE)
        df = subset(df, select = -c(1)) # drop the first column (patient ID)
        df[,1][df[,1] == "M"] = 1
        df[,1][df[,1] == "B"] = 0 # relabel binary class to be {0, 1} -> not {-1,1}, since we are talking about logistic loss
        df[,1] = as.numeric(df[,1])
        names(df)[names(df) == 'V2'] = 'y'
    }
    # banknote authentication
    if (v == 2) {
        suppressMessages(require(nonet))
        data(banknote_authentication)
        df = data.frame(banknote_authentication)
        df = df[, c(5,1,2,3,4)]
        names(df)[names(df) == 'class'] = 'y'
    }
    # room occupancy
    if (v == 3) {
        df = read.csv("occupancy_data.txt", header = TRUE)
        df = subset(df, select = -c(1)) # drop the first column (time)
        df = df[, c(6,1,2,3,4,5)]
        names(df)[names(df) == 'Occupancy'] = 'y'
    }
    if (isHinge) { # is Hinge loss
        df[,1][df[,1] == 0] = -1
    }
    return(df)
}

# test code
# d = load.data(3)