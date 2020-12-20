##############################################################
# Sampling Function.
# Inputs: data.list
#         func.param
# Returns: list of flags for each component
##############################################################

samplingFuncSiteFrac = function(data.list, otherArgs) {
  #bagFrac = 0.5
  bagFrac = otherArgs$bag.frac
  occCovars = data.list$d1
  detCovars = data.list$d2
  index = otherArgs$rel
  nSites = dim(occCovars)[1]
  nObs = dim(detCovars)[1]

  occSample = runif(nSites)<bagFrac
  detSample = NULL
  for (i in 1:nSites) {
    if (occSample[i]==1) {
      detSample[which(index==i)] = TRUE
    } else {
      detSample[which(index==i)] = FALSE
    } # if/else
  } # i
  dim(detSample) = dim(otherArgs$y)

  ret = list(sample1=occSample,sample2=detSample)
  return(ret)
}

#########################################################################
# Gradient Function.  (derivatives of loss function at each data point)
# Inputs: icomp (which component, 1 for occ, 2 for det)
#         predictions (from grt)
#         otherArgs (other arguments: nsite, y, rel)
#########################################################################

gradientFunc <- function(icomp, predictions, bag.flag, otherArgs) { 
  nsite = otherArgs$nsite
  y = otherArgs$y
  rel = otherArgs$rel

  F <- predictions[[1]]
  G <- predictions[[2]]

  sitesToUse = bag.flag$sample1
  obsToUse = bag.flag$sample2

  if (icomp == 1) { # calculate gradient for component 1 (occupancy)
    gradient <- rep(0, nsite)
    for (isite in 1:nsite) { 
      if (sitesToUse[isite]==1) {
        prodTerm <- getProdTerm(y, G, isite, rel)
        identTerm <- getIdentTerm(y, isite, rel)

        psi <- logistic(F[isite])
        gradient[isite] <- (prodTerm - identTerm) * psi * (1 - psi) /
                           (psi * prodTerm + (1 - psi) * identTerm)
      } # if
    } # end for isite
    return(gradient)
  } else if (icomp == 2) { # calculate gradient for component 2 (detection)
    gradient <- rep(0, length(y))
    for (tau in 1:length(y)) {
      if (obsToUse[tau]==1) {
        isite <- rel[tau]
        prodTerm <- getProdTerm(y, G, isite, rel)
        identTerm <- getIdentTerm(y, isite, rel)
        psi <- logistic(F[isite])
        p.tau <- logistic(G[tau])
        gradient[tau] <- prodTerm * psi * (y[tau] - p.tau) /
                         (psi * prodTerm + (1 - psi) * identTerm)
      } # if
    } # end for tau
    return(gradient)
  } else {
    stop("There are only two components in this model!")
  } # end if
}

####################################################
# Intermediate function.  Calculates product term.
# Inputs: y (det/non-det histories)
#         G (current p predictions from grt)
#         isite (which site)
#         rel (index/relation structure)
####################################################

getProdTerm <- function(y, G, isite, rel) { 
  yi <- y[rel == isite]
  gi <- G[rel == isite]
  nobs <- length(yi)
  prodTerm <- 1
  for (t in 1:nobs) {
    if (yi[t] == 1) {
      prodTerm <- prodTerm * logistic(gi[t])
    } else if (yi[t] == 0) {
      prodTerm <- prodTerm * (1 - logistic(gi[t]))
    } else {
      stop("y is not 0 or 1")
    }
  }
  return(prodTerm)
}

###########################################################
# Utility function.  Calculates logistic function.
# Inputs: x (numeric)
###########################################################

logistic <- function(x) {return(1 / (1 + exp(-x)))}

###########################################################
# Intermediate function.  Calculates delta/identity term.
# Inputs: y (det/non-det histories)
#         isite (which site)
#         rel (index/relation structure)
###########################################################

getIdentTerm <- function(y, isite, rel) { 
  yi <- y[rel == isite]
  identTerm <- as.integer(all(yi == 0))
  return(identTerm)
}

########################################################
# Step length function - shrinkage.
# Inputs: icomp (which component)
#         predictions (current grt predictions)
#         otherArgs (other args, including shrinkage)
########################################################

shrinkageStepLength <- function(icomp, predictions, regression, bag.flag,
                                otherArgs) { 
  shrinkage <- otherArgs$shrinkage
  return(shrinkage)
}

#####################################################################
# Loss Function.  (negative log-likelihood of site-occupancy model)
# Inputs: predictions (from grt)
#         bag.flag (list of sample indicators)
#         otherArgs (other arguments: nsite, y, rel)
#####################################################################

lossFunc <- function(predictions,bag.flag,otherArgs) { 
  nsite = otherArgs$nsite
  y = otherArgs$y
  rel = otherArgs$rel

  F <- predictions[[1]]
  G <- predictions[[2]]

  sitesToUse = bag.flag$sample1
  obsToUse = bag.flag$sample2

  sums <- array(0,c(nsite,1))
  for (isite in 1:nsite) {
    prodTerm <- getProdTerm(y, G, isite, rel)
    identTerm <- getIdentTerm(y, isite, rel)
    psi <- logistic(F[isite])
    sums[isite] <- log(psi * prodTerm + (1 - psi) * identTerm)
  } # end for isite

  sums = -sums
  in.bag = mean(sums[which(sitesToUse==1)])
  out.bag = mean(sums[which(sitesToUse==0)])

  return(c(in.bag,out.bag))
}

###############################################################
# Prediction Functions.  (calculates inference probabilities)
# Inputs: predictions (current grt predictions)
#         rel (index/relation structure)
###############################################################

predict_func <- function(predictions, rel, save.flag=FALSE, save.path="",
                         nTrees=0) { 
  occProbs = logistic(predictions[[1]])
  detProbs = logistic(predictions[[2]])
  obsProbs = occProbs[rel$rel]*detProbs
  result = obsProbs

  if (save.flag) {
		write.table(occProbs, paste(save.path,"Pred_occProbs.csv",sep=""),
                row.names=F, col.names=F, sep=",")
		write.table(detProbs, paste(save.path,"Pred_detProbs.csv",sep=""),
                row.names=F, col.names=F, sep=",")
		write.table(result, paste(save.path,"Pred_obsProbs.csv",sep=""),
                row.names=F, col.names=F, sep=",")
  }
  return(result)
}

predict_funcOCC <- function(predictions, rel) { 
  occProbs = logistic(predictions[[1]])
  return(occProbs)
}

predict_funcDET <- function(predictions, rel) { 
  detProbs = logistic(predictions[[2]])
  return(detProbs)
}