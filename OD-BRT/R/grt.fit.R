grt.fit <- function(nComps,
                    compInputs,
                    lossFn,
                    gradFn,
                    otherInputs=NULL,
                    init=NULL,
										depthFn,
                    stepLengthFn,
                    sampleFn=NULL,
                    nTrees=1000,
                    minObs=10,
                    conjGradFlag=TRUE,
                    keepDataFlag=TRUE,
                    verboseFlag=FALSE
                   )
{
  ########
  # utility function:
  checkFlag <- function(bagFlag, nComps, regData) {
    if (!is.list(bagFlag) || length(bagFlag) != nComps) {
      stop("Flag for random data split is not a list,
           or it does not have nComps components.")
    }
    for (icomp in 1:nComps) {
      if (length(bagFlag[[icomp]]) != regData[[icomp]]$nrows) {
        stop("Flag for random data split in component ", icomp,
             " does not has equal number of rows as data.")
      }
    }
  }
  ########

  ########
  # check arguments
  if (missing(nComps) || missing(compInputs) || missing(gradFn) ||
      missing(stepLengthFn)) {
    stop("Some essential argument is missing.")
  }

  if(!is.list(compInputs) || length(compInputs)!= nComps) {
    stop("Please make sure compInputs is a list and has ", nComps,
         " elements.")
  }

  if(!is.null(init) && (!is.list(init) || (length(init)!=nComps))) {
    stop("Please make sure init is a list and has ", nComps, " elements.")
  }
  if (is.null(init)) {
    init <- vector("list", nComps)
    for (icomp in 1:nComps) {
      init[[icomp]] <- rep(0, nrow(compInputs[[icomp]]))
    }
  }
  ########

  ########
  # model setup and data checks
  model <- list()
  model$nComps <- nComps
  model$nTrees <- nTrees
  model$varTypes <- vector("list", nComps)
  model$varLevels <- vector("list", nComps)
  model$varNames <- vector("list", nComps)
  model$data <- NULL

  formattedData=list(nComps)
  for (icomp in 1:nComps) {
    x = compInputs[[icomp]]
    varNames <- colnames(x)

    cRows <- nrow(x)
    cCols <- ncol(x)
    # check dataset size
    if(cRows <= 2*minObs + 1) {
      stop("Dataset size is too small : cRows<= 2 * minObs + 1,
            in component", icomp)
    }
    if(cRows != length(init[[icomp]])) {
      stop("Initial value does not have same rows as data in component ",
           icomp, ". Data has ", cRows, " rows while init has ",
           length(init[[icomp]]), " rows.")
    }

    # setup variable types
    varType <- rep(0,cCols)
    varLevels <- vector("list",cCols)
    for(i in 1:length(varType)) {
      if(all(is.na(x[,i]))) {
        stop("variable ",i,": has only missing values.")
      }
      if(is.ordered(x[,i])) {
        varLevels[[i]] <- levels(x[,i])
        x[,i] <- as.numeric(x[,i])-1
        varType[i] <- 0
      }
      else if(is.factor(x[,i])) {
        if(length(levels(x[,i]))>1024)
          stop("Variable ", i, ": has ", length(levels(x[,i])),
               " levels, more than grt can handle.")
        varLevels[[i]] <- levels(x[,i])
        x[,i] <- as.numeric(x[,i])-1
        varType[i] <- max(x[,i],na.rm=TRUE)+1
      }
      else if(is.numeric(x[,i])) {
        varLevels[[i]] <- quantile(x[,i],prob=(0:10)/10,na.rm=TRUE)
      }
      else {
        stop("variable ",i,": is not of type numeric, ordered, or factor.")
      }
    }
    model$varTypes[[icomp]] <- varType
    model$varLevels[[icomp]] <- varLevels
    model$varNames[[icomp]] <- varNames

    # create index upfront... subtract one for 0 based order
    xOrder <- apply(x,2,order,na.last=FALSE)-1
    xOrder <- as.vector(data.matrix(xOrder))

    x <- as.vector(data.matrix(x))
    newData = list(x=x, nrows=as.integer(cRows), ncols=as.integer(cCols),
                   varType=as.integer(varType), xOrder=as.integer(xOrder))
    formattedData[[icomp]] = newData
  } # icomp

  if (keepDataFlag) {
    model$data <- formattedData
    model$otherInputs <- otherInputs
    model$compInputs <- compInputs
  }
  ########

  ########
  # set up tree sequences
  treeSeqs <- vector("list", nComps)
  predictions <- vector("list", nComps)
  allFlag <- vector("list", nComps)

  for (icomp in 1:nComps) {
    treeSeqs[[icomp]] <- vector("list", nTrees)
    predictions[[icomp]] <- rep(0, times=formattedData[[icomp]]$nrows)
    allFlag[[icomp]] <- rep(TRUE, times=formattedData[[icomp]]$nrows)
  } # icomp
  trainLoss <- rep(0, times=nTrees)
  oobLoss <- rep(0, times=nTrees)

  parameters=list(depth=as.integer(depthFn), n.minobs=as.integer(minObs),
                  step.length=as.double(1))

  gradient <- vector("list", nComps)
  thisTreeOutput <- vector("list", nComps)
  ########

  ########
  # iterate over trees
  for (it in 1:nTrees) {
    if ((it %% 100)==0) { print(paste("Iteration ",it,sep="")) }

    if (!is.null(sampleFn)) {
      bagFlag <- sampleFn(compInputs, otherInputs)
      checkFlag(bagFlag, nComps, formattedData)
    }
    else
      bagFlag <- allFlag

    if (conjGradFlag) {  ######## conjGradFlag=TRUE
      for (icomp in 1:nComps) {
        if (it == 1) {
          gradient[[icomp]] <- gradFn(icomp, init, bagFlag, otherInputs)
        }
        else {
          gradient[[icomp]] <- gradFn(icomp, predictions, bagFlag, otherInputs)
        }
        thisTreeOutput[[icomp]] <- as.double(rep(0, length(gradient[[icomp]])))

        ret <- TRUE
        ret <- .Call("grt_tree", treeSeqs[[icomp]],
                                 it=as.integer(it-1),
                                 parameters,
                                 bagFlag[[icomp]],
                                 formattedData[[icomp]],
                                 gradient[[icomp]],
                                 thisTreeOutput[[icomp]],
                                 PACKAGE="grt")
        if (!ret)
          stop("Error in gradient tree training.")

        step <- stepLengthFn(icomp, predictions, thisTreeOutput, bagFlag,
                             otherInputs)
        treeSeqs[[icomp]][[it]][[3]] <- step
        predictions[[icomp]] <- predictions[[icomp]] +
                                thisTreeOutput[[icomp]] * step
      }
    }
    else {  ######## conjGradFlag=FALSE
      for (icomp in 1:nComps) {
        if (it == 1) {
          gradient[[icomp]] <- gradFn(icomp, init, bagFlag, otherInputs)
        }
        else {
          gradient[[icomp]] <- gradFn(icomp, predictions, bagFlag, otherInputs)
        }
        thisTreeOutput[[icomp]] <- as.double(rep(0, length(gradient[[icomp]])))
      }

      ret <- .Call("grt_iterate", treeSeqs,
                                  it=as.integer(it-1),
                                  parameters,
                                  bagFlag,
                                  formattedData,
                                  gradient,
                                  thisTreeOutput, PACKAGE="grt")
      if (!ret)
        stop("Error in gradient tree training iteration.")

      for (icomp in 1:nComps) {
        step <- stepLengthFn(icomp, predictions, thisTreeOutput, bagFlag,
                             otherInputs)
        treeSeqs[[icomp]][[it]][[3]] <- step
        predictions[[icomp]] <- predictions[[icomp]]
                                + thisTreeOutput[[icomp]] * step
      }
    }

    iter.loss <- lossFn(predictions, bagFlag, otherInputs)
    trainLoss[it] <- iter.loss[1]
    if (length(iter.loss >= 2)) {
      oobLoss[it] <- iter.loss[2]
    }
    if (verboseFlag) {
      cat("In iteration", it, "\n")
      cat("Training loss becomes", trainLoss[it], "\n")
      cat("OOB loss becomes", oobLoss[it], "\n")
      print(paste("Gradients",gradient,"\n"),sep="")
      print(paste("Predictions",predictions,"\n"),sep="")
    }
  } # it
  ########

  ########
  # bundle result and return
  model$treeSeqs <- treeSeqs
  model$trainLoss <- trainLoss
  model$oobLoss <- oobLoss
  model$gradFn <- gradFn
  model$stepLengthFn <- stepLengthFn
  model$lossFn <- lossFn
  model$sampleFn <- sampleFn
  model$depthFn <- depthFn
  model$conjGradFlag <- conjGradFlag
  model$keepDataFlag <- keepDataFlag
  model$minObs <- minObs
  return(model)
  ########
}