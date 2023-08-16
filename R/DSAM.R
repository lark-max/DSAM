#' @title Default parameter list
#' @description
#' The list of parameters needs to be set by the user, each with a default value.
#'
#' \describe{
#' \item{include.inp}{Boolean variable that determines whether the input vectors should be included during the Euclidean distance calculation. The default is \code{TRUE}.  }
#' \item{seed}{Random number seed. The default is \code{1000}.  }
#' \item{sel.alg}{A string variable that represents the available data splitting algorithms including \code{"SOMPLEX"}, \code{"MDUPLEX"}, \code{"DUPLEX"}, \code{"SBSS.P"}, \code{"SS"} and \code{"TIMECON"}. The default is \code{"MDUPLEX"}.  }
#' \item{prop.Tr}{The proportion of data allocated to the training subset, where the default is \code{0.6}.  }
#' \item{prop.Ts}{The proportion of data allocated to the test subset, where the default is \code{0.2}.  }
#' \item{Train}{A string variable representing the output file name for the training data subset. The default is \code{"Train.txt"}.  }
#' \item{Test}{A string variable representing the output file name for the test data subset. The default is \code{"Test.txt"}.  }
#' \item{Validation}{A string variable representing the output file name for the validation data subset. The default is \code{"Valid.txt"}.  }
#' \item{loc.calib}{Vector type: When sel.alg = "TIMECON", the program will select a continuous time-series data subset from the original data set, where the start and end positions are determined by this vector, with the first and the second value representing the start and end position in percentage of the original dataset. The default is \code{c(0,0.6)}, implying that the algorithm selects the first 60% of the data from the original dataset.  }
#' \item{writeFile}{Boolean variable that determines whether the data subsets need to be output or not. The default is \code{FALSE}.  }
#' \item{showTrace}{Boolean variable that determines the level of user feedback. The default is \code{FALSE}.   }
#' }
#'
#' @return None
#'
par.default <- function(){
  list(
    include.inp = TRUE,
    seed = 1000,
    sel.alg = "MDUPLEX",
    prop.Tr = 0.6,
    prop.Ts = 0.2,
    Train = "Train.txt",
    Test = "Test.txt",
    Validation = "Valid.txt",
    loc.calib = c(0,0.6),
    writeFile = FALSE,
    showTrace = FALSE
  )
}


#' @title Get the AUC value between two datasets
#' @description
#' This function calls \code{[kohonen]{xgboost}} to train the classifier, followed by calculating the similarity between the two given datasets. The return value is a AUC index, ranging between 0 and 1, where the AUC is closer to 0.5, the more similar the two data sets is.
#'
#' @param data1 Dataset 1, the data type must be numeric, matrix or Data.frame.
#' @param data2 Dataset 2, the data type must be numeric, matrix or Data.frame.
#'
#' @return Return the AUC value.
#'
#' @importFrom xgboost xgb.DMatrix xgboost
#' @importFrom pROC auc
#' @importFrom Matrix Matrix
#' @importFrom stats predict
#' @importFrom caret createFolds
#'
#' @export
#'
getAUC <- function(data1, data2){
  # Check data format
  if(!mode(data1) %in% c("numeric","matrix","data.frame")){
    stop(
      paste("[Error]:Invalid input data format for the parameter called: ",deparse(substitute(data1)),"!",sep="")
    )
  }
  if(!mode(data2) %in% c("numeric","matrix","data.frame")){
    stop(
      paste("[Error]:Invalid input data format for the parameter called: ",deparse(substitute(data2)),"!",sep="")
    )
  }

  # Check data integrity
  if(sum(is.na(data1))!=0){
    stop(
      paste("[Error]:Missing values in the input data:",deparse(substitute(data1)),"!",sep="")
    )
  }
  if(sum(is.na(data2))!=0){
    stop(
      paste("[Error]:Missing values in the input data:",deparse(substitute(data2)),"!",sep="")
    )
  }

  # Force convert data type to matrix when the type is numeric
  if(mode(data1) %in% "numeric"){
    data1 <- as.matrix(data1)
  }
  if(mode(data2) %in% "numeric"){
    data2 <- as.matrix(data2)
  }

  # Check the dimensions are consistent between the two input datasets
  if(ncol(data1) != ncol(data2)){
    stop(
      "[Error]:Inconsistent dimensions of the two input datasets!"
    )
  }

  # Data preprocessing
  data <- as.data.frame(rbind(data1,data2))
  data$status <- c(rep(1,nrow(data1)),rep(0,nrow(data2)))

  splitNum <- 10
  auc_value <- c()

  # Set K folds
  folds <- caret::createFolds(y=data$status, k=splitNum)

  # The classifier is trained for each fold and the auc value is calculated
  for(i in 1:splitNum){
    train <- data[-folds[[i]],]
    test <- data[folds[[i]],]

    # Data preprocessing
    trainData <- data.matrix(train[,-ncol(train)])
    trainData <- Matrix::Matrix(trainData,sparse = T)
    dtrain <- xgboost::xgb.DMatrix(data = trainData, label = train$status)
    testData <- data.matrix(test[,-ncol(test)])
    testData <- Matrix::Matrix(testData,sparse = T)
    dtest <- xgb.DMatrix(data = testData, label = test$status)

    # xgboost classifier
    model <- xgboost::xgboost(data = dtrain,verbosity = 0, max_depth=6, eta=0.5,nrounds=100,
                     objective='binary:logistic', eval_metric = 'auc', verbose = 0)

    pre_xgb = round(stats::predict(model,newdata = dtest))

    auc_value = c(auc_value, as.numeric(pROC::auc(test$status,pre_xgb,levels=c(1,0),direction=">")))
  }
  return(mean(auc_value))
}


#' @title Select specific split data
#' @description
#' Built-in function: This function decides whether to process the input dataset according to the parameter \code{include.inp}. If TRUE, this function removes Column 1 of the input dataset; otherwise, it returns the Column N of the data set.
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the par.default function.
#'
#' @return Returns a matrix for subsequent calculations.
#'
selectData <- function(data,control){
  if(control)
    return.data = as.matrix(data[,-1]) # Remove the index column
  else
    return.data = as.matrix(data[,ncol(data)]) # Only choose the Output column
  return(return.data)
}


#' @title Standardized data
#' @description
#' Built-in function: This function is used to standardize the data.
#'
#' @param data The dataset should be of type matrix or Data.frame and contain only the input and output vectors.
#'
#' @return Return a matrix with normalized data.
#'
standardise <- function(data){
  if(!is.matrix(data) && !is.data.frame(data)){
    stop(
      "[Error]:Invalid input data format, should be matrix or dataframe"
    )
  }
  if(ncol(data) < 2 || nrow(data) < 2)
    return(as.matrix(data))
  else
    return(as.matrix(apply(data,2,scale)))
}


#' @title Initial sampling of DUPLEX
#' @description
#' Built-in function: The initial sampling function of DUPLEX algorithm, aimed to obtain the two data points with the farthest Euclidean distance from the original data set and assign them to the corresponding sampling subset.
#'
#' @param split.info A list containing relevant sampling information such as the original dataset and three sample subsets.
#' @param choice The variable must be one name of the three sample subsets contained in split.info, according to which the function assigns the current two data points to the specific sampling subset.
#'
#' @importFrom stats dist
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
DP.initialSample <- function(split.info, choice){
  if(length(split.info$index) < 2)
    return(split.info)

  # Generate euclidean distance matrix
  distMat <- as.matrix(stats::dist(split.info$data[split.info$index,],method = "euclidean"))
  max.points <- which(distMat == max(distMat), arr.ind = T)

  eval(parse(text = paste("split.info$",choice,"=c(split.info$",choice,
                          ",split.info$index[c(max.points[1,1],max.points[1,2])])",sep="")))
  eval(parse(text = paste("split.info$index = split.info$index[c(-max.points[1,1],-max.points[1,2])]")))
  return(split.info)
}


#' @title Repeat sampling of DUPLEX
#' @description
#' Built-in function: The cyclic sampling function of DUPLEX algorithm that takes the two data points farthest from the current sampling set and assigns them to the corresponding sampling subset.
#'
#' @param split.info A list containing relevant sampling information such as the original dataset and three sample subsets.
#' @param choice The variable must be one name of the three sample subsets contained in split.info, according to which the function assigns the current two data points to the specific sampling subset.
#'
#' @importFrom stats dist
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
DP.reSample <- function(split.info, choice){
  if(length(split.info$index) < 2){
    eval(parse(text = paste("split.info$",choice,"=c(split.info$",choice,
                            ",split.info$index)",sep="")))
    split.info$index <- c()
    return(split.info)
  }

  originSet <- as.matrix(split.info$data[split.info$index,])
  sampleSet <- NULL
  eval(parse(text = paste("sampleSet <- as.matrix(split.info$data[split.info$",choice,",])",sep="")))
  len.org <- nrow(originSet)
  len.sam <- nrow(sampleSet)
  mergeSet <- rbind(originSet,sampleSet)
  len.mge <- len.org + len.sam

  # Generate euclidean distance matrix
  # distMat <- as.matrix(stats::dist(mergeSet,method = "euclidean"))
  distMat <- matrix(apply(mergeSet,1,crossprod),
                    nrow = len.mge, ncol = len.mge)
  distMat <- distMat + t(distMat) - 2*tcrossprod(mergeSet)

  # Generate Single-Linkage distance vector
  singleLinkageDist <- apply(distMat[(len.org+1):(len.org+len.sam),1:len.org],2,min)

  max.point <- which.max(singleLinkageDist)
  singleLinkageDist[max.point] = -1.0e+6
  secmax.point <- which.max(singleLinkageDist)

  eval(parse(text = paste("split.info$",choice,"=c(split.info$",choice,
                          ",split.info$index[c(max.point,secmax.point)])",sep="")))
  eval(parse(text = paste("split.info$index = split.info$index[c(-max.point,-secmax.point)]")))
  return(split.info)
}


#' @title Check whether the sample set is full
#' @description
#' Built-in function: This function includes four arguments,
#' where the first one contains the information of the original dataset as well as the three subsets,
#' and the remaining three augments are the maximum sample sizes for the training, test and validation subsets respectively.
#'
#' @param split.info List type, which contains the original data set, three sampling subsets, termination signal and other relevant sampling information.
#' @param num.train The number of training data points specified by the user.
#' @param num.test The number of test data points specified by the user.
#' @param num.valid The number of validation data points specified by the user.
#'
#' @return A list with sampling information.
#'
checkFull <- function(split.info, num.train, num.test, num.valid){
  res <- list(ini.info = split.info)
  if(!(length(split.info$trainKey) < num.train) & !(length(split.info$testKey) < num.test)){
    res$ini.info$validKey = c(res$ini.info$validKey, res$ini.info$index)
    res$ini.info$index = c()
    res$signal = TRUE
  }else if(!(length(split.info$trainKey) < num.train) & !(length(split.info$validKey) < num.valid)){
    res$ini.info$testKey = c(res$ini.info$testKey, res$ini.info$index)
    res$ini.info$index = c()
    res$signal = TRUE
  }else if(!(length(split.info$testKey) < num.test) & !(length(split.info$validKey) < num.valid)){
    res$ini.info$trainKey = c(res$ini.info$trainKey, res$ini.info$index)
    res$ini.info$index = c()
    res$signal = TRUE
  }else{
    res$signal = FALSE
  }
  return(res)
}


#' @title Self-organized map clustering
#' @description
#' Built-in function: This function performs clustering for a given dataset by calling the \code{[kohonen]{som}} function from a “kohonen” package.
#'
#' @param data The dataset in matrix or data.frame, containing only input and output vectors, but with no subscript vector.
#'
#' @return Return a data list of clustering neurons in the SOM network.
#' @importFrom kohonen som somgrid
#'
somCluster <- function(data){
  # Basic parameters for som cluster
  neuron.num = round(2*sqrt(nrow(data)))
  neuron.col = round(sqrt(neuron.num / 1.6))
  neuron.row = round(1.6 * neuron.col)
  neuron.info <- list(
    neuron.num = neuron.row * neuron.col,
    neuron.row = neuron.row,
    neuron.col = neuron.col
  )
  # som train, need loading the package: kohonen
  som.model <- kohonen::som(data, grid = kohonen::somgrid(neuron.info$neuron.row, neuron.info$neuron.col,
                                        "hexagonal"))
  # summary
  neuron.info$neuron.cluster <- list()
  for(i in 1:neuron.info$neuron.num){
    neuron.info$neuron.cluster[[i]] = which(som.model$unit.classif == i)
  }
  return(neuron.info)
}


#' @title Get sampling number of each SOM neuron
#' @description
#' Built-in function: Calculates the maximum number of samples of each subset in each neuron within the SOM network based on the sampling ratio specified by the user.
#'
#' @param som.info The list contains information about the SOM network, including the total number of neurons, the number of rows, and the set of data points within each neuron.
#' @param control User-defined parameter list, where each parameter definition refers to the par.default function.
#'
#' @return This function return a list containing three vectors Tr,Ts and Vd, the length of which is the same as the number of neurons. Tr,Ts and Vd vectors record the specified amount of data that need be obtained for the Training, Test and Validation subset in each neuron respectively.
#'
getSnen <- function(som.info, control){
  sampleNum.eachNeuron = list()
  sampleNum.eachNeuron$Tr <- sampleNum.eachNeuron$Ts <- sampleNum.eachNeuron$Vd <- c()
  for(i in 1:som.info$neuron.num){
    sampleNum.eachNeuron$Tr[i] = round(length(som.info$neuron.cluster[[i]])* control$prop.Tr)
    sampleNum.eachNeuron$Ts[i] = round(length(som.info$neuron.cluster[[i]])* control$prop.Ts)
    sampleNum.eachNeuron$Vd[i] = length(som.info$neuron.cluster[[i]]) -
      sampleNum.eachNeuron$Tr[i] - sampleNum.eachNeuron$Ts[i]
  }
  return(sampleNum.eachNeuron)
}


#' @title 'DSAM' - Time-consecutive algorithm
#' @description
#' This function selects a time-consecutive data from the original data set as the calibration (training and test) subset, and the remaining data is taken as the evaluation subset.
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the \code{\link{par.default}} function.
#'
#' @return Return the calibration and validation subsets.
#'
TIMECON <- function(data, control){
  len <- nrow(data)
  start.point = 1 + len * control$loc.calib[1]
  end.point = len * control$loc.calib[2]

  if(start.point<0 | start.point > len | end.point < 0 | end.point > len | end.point < start.point)
    stop(
      "[Error]:Parameter loc.calib is incorrect, should be set to [0,1]"
    )

  return(list(Calibration = data[start.point:end.point,],
              Validation = data[-(start.point:end.point),]))
}


#' @title Core function of SS sampling
#' @description
#' Built-in function: This function performs the SS algorithm.
#'
#' @param index A subscript vector whose subscript corresponds to the output vector of the data point sorted in an ascending order.
#' @param prop The sampling ratio, with the value ranging between 0 and 1.
#'
#' @importFrom stats runif
#' @return Return a vector containing the subscript of the sampled data points.
#'
SSsample <- function(index, prop){
  sampleVec <- c()
  interval <- 1/prop
  loc <- stats::runif(1,0,32767) %% ceiling(interval)
  while(loc <= length(index)){
    k <- ceiling(loc)
    sampleVec <- c(sampleVec, index[k])
    loc = loc + interval
  }
  return(sampleVec)
}


#' @title Get the remain unsampled data after \code{\link{SSsample}}
#' @description
#' Built-in function: This function is used in the semi-deterministic SS algorithm, and it contains two parameters X and Y, both of which are in an increased order. All data points in X vector that have not appeared in Y vector will be recorded and returned by this function.
#'
#' @param X A vector that needs to be sampled.
#' @param Y A vector with data samples from X.
#'
#' @return A vector containing the remaining data that are not in Y.
#'
remainUnsample <- function(X, Y){
  remainData <- c()
  ix <- iy <- 1
  while(ix <= length(X)){
    if(iy > length(Y))
      break
    if(X[ix] == Y[iy])
      iy = iy + 1
    else
      remainData  = c(remainData, X[ix])
    ix = ix + 1
  }
  if(ix <= length(X))
    remainData = c(remainData, X[ix:length(X)])
  return(remainData)
}


#' @title 'DSAM' - SS algorithm
#' @description
#' The systematic stratified (SS) is a semi-deterministic method, with details given in Zheng et al. (2018).
#'
#' @references
#' Zheng, F., Maier, H.R., Wu, W., Dandy, G.C., Gupta, H.V. and Zhang, T. (2018) On Lack of Robustness in Hydrological Model Development Due to Absence of Guidelines for Selecting Calibration and Evaluation Data: Demonstration for Data‐Driven Models. Water Resources Research 54(2), 1013-1030.
#'
#' @param data The type of data set to be divided should be matrix or Data.frame, and the data format is as follows: The first column is a subscript vector, which is used to mark each data point (each row is regarded as a data point); Columns 2 through N-1 are the input vectors, and columns N (the last) are the output vectors.
#' @param control User-defined parameter list, where each parameter definition refers to the \code{\link{par.default}} function.
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
SS <- function(data, control){
  # Basic parameters
  set.seed(control$seed)
  num.total = nrow(data)
  num.train = round(num.total*control$prop.Tr)
  num.test = round(num.total*control$prop.Ts)
  num.valid = num.total- (num.train + num.test)

  data.index <- 1:num.total

  # Firstly get the output variable list
  outputVec <- data[,ncol(data)]

  # The data are first ordered along the output variable dimension in increasing order
  data.index <- data.index[order(outputVec),drop = FALSE]

  Prop.calib = control$prop.Tr + control$prop.Ts
  trainKey <- testKey <- validKey <- c()

  # The data is firstly divided into two parts,
  # which need to be sampled and those that do not need to be sampled
  sampleKey <- SSsample(data.index,Prop.calib)

  # The unsampled data are allocated to validating subset
  validKey <- remainUnsample(data.index, sampleKey)

  # Then split sample into systematic testing and training sets
  testKey <- SSsample(sampleKey, control$prop.Ts / Prop.calib)
  trainKey <- remainUnsample(sampleKey, testKey)

  if(control$showTrace){
    message("SS sampling complete!")
  }
  return(list(Train = data[trainKey,], Test = data[testKey,],
              Validation = data[validKey,]))
}


#' @title 'DSAM' - SBSS.P algorithm
#' @description
#' SBSS.P algorithm is a stochastic algorithm. It obtains data subsets through uniform sampling in each neuron after clustering through SOM neural network, with details given in May et al. (2010).
#'
#' @references
#' May, R. J., Maier H. R., and Dandy G. C.(2010), Data splitting for artificial neural networks using SOM-based stratified sampling, Neural Netw, 23(2), 283-294.

#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the par.default function.
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
SBSS.P <- function(data, control){
  # Basic parameters
  set.seed(control$seed)
  num.total = nrow(data)
  num.train = round(num.total*control$prop.Tr)
  num.test = round(num.total*control$prop.Ts)
  num.valid = num.total- (num.train + num.test)

  # deal with data
  select.data = selectData(data, control$include.inp)
  select.data.std = standardise(select.data)

  # SOM cluster
  som.info <- somCluster(select.data.std)

  # Determine the amount of sampled data for each neuron
  sampleNum.eachNeuron <- getSnen(som.info, control)

  # Sampling
  trainSet <- testSet <- ValidSet <- c()
  if(control$showTrace){
    message("Total neuron: ", som.info$neuron.num)
  }

  # Create information list
  split.info <- list(
    trainKey = c(),
    testKey = c(),
    validKey = c()
  )

  # Sampling from each neuron
  for(i in 1:som.info$neuron.num){
    if(control$showTrace){
      message("sampling on neuron: ",i)
    }
    # Sampling for training dataset
    if(length(som.info$neuron.cluster[[i]]) > sampleNum.eachNeuron$Tr[i]){
      randomSample.index <- sample(c(1:length(som.info$neuron.cluster[[i]])), sampleNum.eachNeuron$Tr[i])
      split.info$trainKey <- c(split.info$trainKey, som.info$neuron.cluster[[i]][randomSample.index])
      som.info$neuron.cluster[[i]] = som.info$neuron.cluster[[i]][-randomSample.index]
    }
    # Sampling for test dataset
    if(length(som.info$neuron.cluster[[i]]) > sampleNum.eachNeuron$Ts[i]){
      randomSample.index <- sample(c(1:length(som.info$neuron.cluster[[i]])), sampleNum.eachNeuron$Ts[i])
      split.info$testKey <- c(split.info$testKey, som.info$neuron.cluster[[i]][randomSample.index])
      som.info$neuron.cluster[[i]] = som.info$neuron.cluster[[i]][-randomSample.index]
    }
    # The remain data points are all allocated to validation dataset
    split.info$validKey <- c(split.info$validKey, som.info$neuron.cluster[[i]])
    som.info$neuron.cluster[[i]] <- NA
  }
  if(control$showTrace){
    message("SBSS.P sampling complete!")
  }
  return(list(Train = data[split.info$trainKey,], Test = data[split.info$testKey,],
              Validation = data[split.info$validKey,]))

}


#' @title 'DSAM' - DUPLEX algorithm
#' @description
#' The deterministic DUPLEX algorithm, with details given in Chen et al. (2022).
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the \code{\link{par.default}} function.
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
DUPLEX <- function(data,control){
  # Basic parameters
  set.seed(control$seed)
  num.total = nrow(data)
  num.train = round(num.total*control$prop.Tr)
  num.test = round(num.total*control$prop.Ts)
  num.valid = num.total- (num.train + num.test)

  # deal with data
  select.data = selectData(data, control$include.inp)
  select.data.std = standardise(select.data)

  # Create information list
  split.info <- list(
    data = select.data.std,
    index = seq(1,num.total,1),
    trainKey = c(),
    testKey = c(),
    validKey = c()
  )

  # Step1: initial sampling
  if(control$showTrace){
    message("Start the initial sampling...")
  }
  if(num.train > 0)
    split.info = DP.initialSample(split.info,"trainKey")
  if(num.test > 0)
    split.info = DP.initialSample(split.info,"testKey")
  if(num.valid > 0)
    split.info = DP.initialSample(split.info,"validKey")
  if(control$showTrace){
    message("Initial sampling successfully!")
    message("Start the loop sampling...")
  }

  # Step2: sampling data through a cyclic sampling pool
  while(length(split.info$index) > 0){
    if(control$showTrace){
      message("Remaining unsampled data: ",length(split.info$index))
    }
    if(length(split.info$trainKey) < num.train)
      split.info = DP.reSample(split.info,"trainKey")
    if(length(split.info$testKey) < num.test)
      split.info = DP.reSample(split.info,"testKey")
    if(length(split.info$validKey) < num.valid)
      split.info = DP.reSample(split.info,"validKey")
    # Check full
    check.res = checkFull(split.info, num.train, num.test, num.valid)
    if(check.res$signal){ # The stop signal is TRUE
      if(control$showTrace){
        message("Two of the datasets are full, and all remaining data is sampled to the other dataset")
      }
      split.info = check.res$ini.info
      break;
    }
  }
  if(control$showTrace){
    message("DUPLEX sampling complete!")
  }
  return(list(Train = data[split.info$trainKey,], Test = data[split.info$testKey,],
              Validation = data[split.info$validKey,]))
}


#' @title 'DSAM' - MDUPLEX algorithm
#' @description
#' This is a modified MDUPLEX algorithm, which is also deterministic, with details given in Zheng et al. (2022).
#'
#' @references
#' Chen, J., Zheng F., May R., Guo D., Gupta H., and Maier H. R.(2022), Improved data splitting methods for data-driven hydrological model development based on a large number of catchment samples, Journal of Hydrology, 613.
#' @references
#' Zheng, F., Chen J., MaierH. R., and Gupta H.(2022), Achieving Robust and Transferable Performance for Conservation‐Based Models of Dynamical Physical Systems, Water Resources Research, 58(5).
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the par.default function.
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
MDUPLEX <- function(data,control){
  # Basic parameters
  set.seed(control$seed)
  num.total = nrow(data)
  num.train = round(num.total*control$prop.Tr)
  num.test = round(num.total*control$prop.Ts)
  num.valid = num.total- (num.train + num.test)

  # deal with data
  select.data = selectData(data, control$include.inp)
  select.data.std = standardise(select.data)

  # Parameters of the basic sampling pool
  poolSize = round(1/min(control$prop.Tr, control$prop.Ts,
                         1-(control$prop.Tr+control$prop.Ts)))
  samplingPool <- list(
    trainSize = round(poolSize * control$prop.Tr),
    testSize = round(poolSize * control$prop.Ts),
    validSize = round(poolSize * (1-control$prop.Tr-control$prop.Ts))
  )

  # Create information list
  split.info <- list(
    data = select.data.std,
    index = seq(1,num.total,1),
    trainKey = c(),
    testKey = c(),
    validKey = c()
  )

  # Step1: initial sampling
  if(control$showTrace){
    message("Start the initial sampling...")
  }
  if(num.train > 0)
    split.info = DP.initialSample(split.info,"trainKey")
  if(num.test > 0)
    split.info = DP.initialSample(split.info,"testKey")
  if(num.valid > 0)
    split.info = DP.initialSample(split.info,"validKey")
  if(control$showTrace){
    message("Initial sampling successfully!")
    message("Start the loop sampling...")
  }

  # Step2: sampling data through a cyclic sampling pool
  while (length(split.info$index)>0) {
    trainSize.cnt = samplingPool$trainSize
    testSize.cnt = samplingPool$testSize
    validSize.cnt = samplingPool$validSize
    if(control$showTrace){
      message("Remaining unsampled data: ",length(split.info$index))
    }
    while(TRUE){
      stopSignal = TRUE
      if(trainSize.cnt!=0 & length(split.info$trainKey) < num.train){
        split.info = DP.reSample(split.info,"trainKey")
        trainSize.cnt = trainSize.cnt - 1
        stopSignal = FALSE
      }
      if(testSize.cnt!=0 & length(split.info$testKey) < num.test){
        split.info = DP.reSample(split.info,"testKey")
        testSize.cnt = testSize.cnt - 1
        stopSignal = FALSE
      }
      if(validSize.cnt!=0 & length(split.info$validKey) < num.valid){
        split.info = DP.reSample(split.info,"validKey")
        validSize.cnt = validSize.cnt - 1
        stopSignal = FALSE
      }
      if(stopSignal)
        break
    }
  }
  if(control$showTrace){
    message("MDUPLEX sampling complete!")
  }
  return(list(Train = data[split.info$trainKey,], Test = data[split.info$testKey,],
              Validation = data[split.info$validKey,]))
}


#' @title 'DSAM' - SOMPLEX algorithm
#' @description
#' SOMPLEX algorithm is a stochastic algorithm, with details given in Chen et al. (2022) and Zheng et al. (2023)
#'
#' @references
#' Chen, J., Zheng F., May R., Guo D., Gupta H., and Maier H. R.(2022), Improved data splitting methods for data-driven hydrological model development based on a large number of catchment samples, Journal of Hydrology, 613.
#'
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the \code{\link{par.default}} function.
#'
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#'
SOMPLEX <- function(data, control){
  # Basic parameters
  set.seed(control$seed)
  num.total = nrow(data)
  num.train = round(num.total*control$prop.Tr)
  num.test = round(num.total*control$prop.Ts)
  num.valid = num.total- (num.train + num.test)

  select.data = selectData(data, control$include.inp)
  select.data.std = standardise(select.data)

  # SOM cluster
  som.info <- somCluster(select.data.std)

  # Determine the amount of sampled data for each neuron
  sampleNum.eachNeuron <- getSnen(som.info, control)

  # Sampling
  trainSet <- testSet <- ValidSet <- c()
  if(control$showTrace){
    message("Total neuron: ", som.info$neuron.num)
  }
  for(i in 1:som.info$neuron.num){
    # Create information list
    split.info <- list(
      data = select.data.std,
      index = som.info$neuron.cluster[[i]],
      trainKey = c(),
      testKey = c(),
      validKey = c()
    )
    if(control$showTrace){
      message("sampling on neuron: ",i)
    }
    if(sampleNum.eachNeuron$Tr[i] > 0)
      split.info = DP.initialSample(split.info,"trainKey")
    if(sampleNum.eachNeuron$Ts[i] > 0)
      split.info = DP.initialSample(split.info,"testKey")
    if(sampleNum.eachNeuron$Vd[i] > 0)
      split.info = DP.initialSample(split.info,"validKey")

    while(length(split.info$index) > 0){
      if(length(split.info$trainKey) < sampleNum.eachNeuron$Tr[i])
        split.info = DP.reSample(split.info,"trainKey")
      if(length(split.info$testKey) < sampleNum.eachNeuron$Ts[i])
        split.info = DP.reSample(split.info,"testKey")
      if(length(split.info$validKey) < sampleNum.eachNeuron$Vd[i])
        split.info = DP.reSample(split.info,"validKey")
    }
    trainSet = c(trainSet, split.info$trainKey)
    testSet = c(testSet, split.info$testKey)
    ValidSet = c(ValidSet, split.info$validKey)
  }
  if(control$showTrace){
    message("SOMPLEX sampling complete!")
  }
  return(list(Train = data[trainSet,], Test = data[testSet,], Validation = data[ValidSet,]))
}


#' @title Main function of data splitting algorithm
#' @description
#' 'DSAM' interface function: The user needs to provide a parameter list before data-splitting.
#' These parameters have default values, with details given in the \code{\link{par.default}} function.
#' Conditioned on the parameter list, this function carries out the data-splitting based on the algorithm specified by the user.
#' The available algorithms include the traditional time-consecutive method (TIMECON), DUPLEX, MDUPLEX SOMPLEX, SBSS.P, SS.
#' The algorithm details can be found in Chen et al. (2022). Note that this package focuses on deals with the dataset with multiple inputs but one output,
#' where this output is used to enable the application of various data-splitting algorithms.
#'
#'
#' @param data The dataset should be matrix or Data.frame. The format should be as follows: Column one is a subscript vector used to mark each data point (each row is considered as a data point); Columns from 2 to N-1 are the input data, and Column N are the output data.
#' @param control User-defined parameter list, where each parameter definition refers to the \code{\link{par.default}} function.
#' @param ... A redundant argument list.
#'
#' @importFrom utils modifyList write.table
#' @return Return the training, test and validation subsets. If the original data are required to be split into two subsets, the training and test subsets can be combined into a single calibration subset.
#' @export
#'
#' @author
#' Feifei Zheng \email{feifeizheng@zju.edu.cn}
#' @author
#' Junyi Chen \email{jun1chen@zju.edu.cn}
#'
#' @references
#' Chen, J., Zheng F., May R., Guo D., Gupta H., and Maier H. R.(2022).Improved data splitting methods for data-driven hydrological model development based on a large number of catchment samples, Journal of Hydrology, 613.
#' @references
#' Zheng, F., Chen J., Maier H. R., and Gupta H.(2022). Achieving Robust and Transferable Performance for Conservation‐Based Models of Dynamical Physical Systems, Water Resources Research, 58(5).
#' @references
#' Zheng, F., Chen, J., Ma, Y.,  Chen Q., Maier H. R., and Gupta H.(2023). A Robust Strategy to Account for Data Sampling Variability in the Development of Hydrological Models, Water Resources Research, 59(3).
#'
#' @examples
#' data("DSAM_test_smallData")
#' res.sml = dataSplit(DSAM_test_smallData)
#'
#' data("DSAM_test_modData")
#' res.mod = dataSplit(DSAM_test_modData, list(sel.alg = "SBSS.P"))
#'
#' data("DSAM_test_largeData")
#' res.lag = dataSplit(DSAM_test_largeData, list(sel.alg = "SOMPLEX"))
#'
dataSplit <- function(data,control = list(),...){
  # Check data format
  if(!is.matrix(data) & !is.data.frame(data)){
    stop(
      "[Error]:Invalid input data format!"
    )
  }
  # Check data integrity
  if(sum(is.na(data))!=0){
    stop(
      "[Error]:Missing values in the input data!"
    )
  }
  # Check parameter list
  stopifnot(is.list(control))
  control <- utils::modifyList(par.default(),control)
  isValid <- names(control) %in% names(par.default())
  if (any(!isValid)) {
    stop(
      "[Error]:Unrecognised options: ",
      toString(names(control)[!isValid])
    )
  }

  # check the sampling proportion
  if(control$prop.Tr > 1 | control$prop.Tr < 0){
    stop(
      "[Error]:Invalid range of parameter 'prop.Tr', should be [0,1]!"
    )
  }
  if(control$prop.Ts > 1 | control$prop.Ts < 0){
    stop(
      "[Error]:Invalid range of parameter 'prop.Ts', should be [0,1]!"
    )
  }
  if(control$prop.Tr + control$prop.Ts > 1){
    stop(
      "[Error]:prop.Tr+prop.Ts should be [0,1]!"
    )
  }


  # Split data
  start.time = Sys.time()
  eval(parse(text = paste("obj=",control$sel.alg,"(data,control)",sep="")))
  obj$func.time = Sys.time() - start.time

  # Output data into txt files
  if(control$writeFile){
    if(control$sel.alg != "TIMECON"){
      utils::write.table(obj$Train,control$Train,row.names = F,col.names = T,sep='\t')
      utils::write.table(obj$Test,control$Test,row.names = F,col.names = T,sep='\t')
      utils::write.table(obj$Validation,control$Validation,row.names = F,col.names = T,sep='\t')
    }else{
      utils::write.table(obj$Calibration,"Calibration.txt",row.names = F,col.names = T,sep='\t')
      utils::write.table(obj$Validation,"Validation.txt",row.names = F,col.names = T,sep='\t')
    }
  }
  return(obj)
}
