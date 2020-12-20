grt.predict <- function(model,
												compInputs,
												predFn=NULL,
												otherInputs=NULL,
												nTrees=model$nTrees,
												save.flag=FALSE,
												save.path="")
{
  nComps <- model$nComps

  if ((!is.list(compInputs)) || (length(compInputs) != model$nComps)) {
    stop("Please make sure compInputs is a list, and has same number of
         elements as number of components in model")
  }

  if (!is.vector(nTrees) || min(nTrees) < 1 || max(nTrees > model$nTrees)) {
    stop("nTrees should be a vector, and its values should be between 1 and ",
         model$nTrees, ".")
  }

  formattedData=vector("list", nComps)
  for (icomp in 1:nComps) {
    x = compInputs[[icomp]]
    cRows <- nrow(x)
    cCols <- ncol(x)

    if (cCols != length(model$varTypes[[icomp]])) {
        stop(paste("Training data has ", length(model$varTypes[[icomp]]),
                   " features, while testing data has ",
                   cCols, "features in component ", icomp, "."))
    }

    for(i in 1:cCols) {
      if(is.factor(x[,i]) != (model$varTypes[[icomp]][i] != 0)) {
        stop(paste("Types of feature mismatch in training and testing data: ",
                   i, "-th feature of the ", icomp, "-th component."))
      }
      if(is.factor(x[,i])) {
        j <- match(levels(x[,i]), model$varLevels[[icomp]][[i]])
        if(any(is.na(j))) {
          stop(paste("New levels for variable ",
               i,": ", levels(x[,i])[is.na(j)],sep=""))
        }
        x[,i] <- as.numeric(x[,i])-1
      }
    }
    # setup variable types
    varType <- model$varTypes[[icomp]]
    # create index upfront... subtract one for 0 based order
    x <- as.vector(data.matrix(x))

    newData=list(x=x, nrows=as.integer(cRows), ncols=as.integer(cCols),
                 varType=as.integer(varType), x.order=NULL)
    formattedData[[icomp]]=newData
  }

  compPred <- .Call("grt_predict",
                     model$treeSeqs,
                     nTrees=as.integer(nTrees - 1),
                     formattedData,
                     PACKAGE="grt")

  for (icomp in 1:model$nComps) {
    compPred[[icomp]] <- matrix(compPred[[icomp]], ncol=length(nTrees))
  }

  if (is.null(predFn))
    return(compPred)

	if (save.flag) {
		finalPred <- predFn(compPred, otherInputs, save.flag, save.path, nTrees)
	} else {
		finalPred <- predFn(compPred, otherInputs)
	}

	return(finalPred)
}