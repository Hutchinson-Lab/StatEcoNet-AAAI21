remove(list=ls())
source("R/grt.fit.R")
source("R/grt.predict.R")
source("utils.R")
source("tune_params.R") # for tuning parameters
require(PRROC)
require(reshape2)
dyn.load("src/grt.so") # load c++ libarary

tuning = FALSE
# Global variables ===========================================================
if (tuning) {
	# Range of tuning parameters
	bagFrac = c(0.1, 1)
  depths = c(2, 10)
	shrinkages = c(0.1, 1)
  nTrees = rep(1:1000)
} else {
	bagFrac = NA
	depths = NA
	shrinkages = NA
	nTrees = rep(1:1000)
}
n.iters <- max(nTrees)
# ============================================================================

# Keep track of loss/accuracy ================================================
# 1. Training performance
train.loss = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
train.auroc = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
train.auprc = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
train.occCorr = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
train.detCorr = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))

# 2. Validation performance
val.loss = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
val.auroc = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
val.auprc = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
val.occCorr = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))
val.detCorr = array(NaN, c(length(depths), length(shrinkages), length(nTrees)),
					dimnames = list(paste(depths), paste(shrinkages), paste(nTrees)))

# 3. Test performance - only with the optimal setting of lr and tc
test.loss = array(NaN, c(length(nTrees)), dimnames=list(paste(nTrees)))
test.auroc = array(NaN, c(length(nTrees)), dimnames=list(paste(nTrees)))
test.auprc = array(NaN, c(length(nTrees)), dimnames=list(paste(nTrees)))
test.occCorr = array(NaN, c(length(nTrees)), dimnames=list(paste(nTrees)))
test.detCorr = array(NaN, c(length(nTrees)), dimnames=list(paste(nTrees)))

# Empty variables
trainDataGlobal <<- NULL
validateDataGlobal <<- NULL
grt.train.inputs.Global <<- NULL
grt.val.inputs.Global <<- NULL
# ============================================================================

trainEvaluteModel <- function(myData, save.path) {
	trainData <- myData$trainData
	validateData <- myData$validateData
	testData <- myData$testData
	grt.train.inputs <- list(d1=trainData$occCovars, d2=trainData$detCovars)
	grt.val.inputs <- list(d1=validateData$occCovars, d2=validateData$detCovars)
	grt.test.inputs <- list(d1=testData$occCovars, d2=testData$detCovars)

  # 1. Find the optimal n.trees
	ptm.learning <- proc.time()
	opts = trainModel(trainData, validateData, grt.train.inputs, grt.val.inputs,
                    save.path)
	elapsed.learning = (proc.time() - ptm.learning)[3]
  print(paste("*** Finding the optimal n.trees (", opts$best.n.iters,
              ") is done. (", round(elapsed.learning,2), "s)", sep=""))

  # 2. Train the optimal model
  ptm.learning <- proc.time()
  final.model = trainOptModel(opts, trainData, grt.train.inputs, save.path,
                              FALSE)
  elapsed.learning = (proc.time() - ptm.learning)[3]
  print(paste("*** Retraining the optimal model is done. (",
              round(elapsed.learning,2), "s)", sep=""))

  # 3. Test the trained model
  ptm.testing <- proc.time()
  test.results = testModel(final.model, opts, testData, grt.test.inputs,
                           save.path, FALSE)
  elapsed.testing = (proc.time() - ptm.testing)[3]
  print(paste("*** Testing the trained model is done. (",
              round(elapsed.testing,2), "s)", sep=""))

	# Reset the parameters for other runs
  bagFrac <<- c(0.1, 1)
  depths <<- c(2, 10)
	shrinkages <<- c(0.1, 1)
  nTrees <<- rep(1:1000)

  return (list(elapse=elapsed.learning, best.nt = opts$best.n.iters,
               test=test.results))
}

trainModel <- function(trainData, validateData, grt.train.inputs,
                       grt.val.inputs, save.path) {
  trainDataGlobal <<- trainData
  validateDataGlobal <<- validateData
  grt.train.inputs.Global <<- grt.train.inputs
  grt.val.inputs.Global <<- grt.val.inputs

	n.site <- nrow(trainData$occCovars)
	n.obs.total <- length(trainData$detHists)

	# Initialize the component functions to be used in the first iteration.
	init.f <- list(g1=runif(n.site), g2=runif(n.obs.total))

  if (tuning) {
		#Bayesian Search Function
		bayes_bound <- list(tc = c(depths[1], depths[length(depths)]),
                        lr = c(shrinkages[1], shrinkages[length(shrinkages)]),
                        bf = c(bagFrac[1], bagFrac[length(bagFrac)]))
		print(bayes_bound)
    bayesian_results <- rBayesianOptimization::BayesianOptimization(
                        bayesian_func, bounds = bayes_bound, init_points = 2,
                        n_iter = 1, acq = "ucb", kappa = 2.576, eps = 0.0,
                        verbose = TRUE)

    depths <<- round(bayesian_results$Best_Par[1])
    shrinkages <<- bayesian_results$Best_Par[2]
    bagFrac <<- bayesian_results$Best_Par[3]
	}

  # Loop over all combinations of tuning parameters:
  for (i in 1:length(depths)) {	# for all interaction depths
    tc = depths[i]
    for (s in 1:length(shrinkages)) {	# for all shrinkages
      shrink = shrinkages[s]
      # [Training] ============================================================
      # Bundle the additional arguments required by the functions that
      #  grt will use (gradientFunc(), shrinkageStepLength(),
      #  lossFunc(), samplingFuncSiteFrac(), and predict_func()).
      func.param <- list(
        nsite = nrow(grt.train.inputs$d1), # number of sites
        y = trainData$detHists, # detection histories
        rel = trainData$index, # correspondence between observations and sites
        shrinkage = shrink, # shrinkage for shrinkageStepLength()
        bag.frac = bagFrac # bagging fraction
      )

      # Fit a GRT model for this combination of parameters:
      model <- grt.fit(
        nComps=2, # here we have two components: occupancy and detection
        compInputs=grt.train.inputs, # training data
        gradFn=gradientFunc, # function calculating the gradient
        stepLengthFn=shrinkageStepLength, # function calculating step length
        lossFn=lossFunc, # function calculating value of the objective function
        sampleFn=samplingFuncSiteFrac, # function for sampling instances
        otherInputs=func.param, # other arguments for functions above
        init=init.f, # initial values for each component function
        nTrees=n.iters, # number of iterations to run (i.e. number of trees)
        depthFn=tc, # interaction depth of trees
        minObs=1, # minimum number of observations in each leaf of each tree
        conjGradFlag=TRUE, # whether to use coordinate gradient
        keepDataFlag=TRUE # whether to keep training data in model
      )

      func.param.train <- list(
        nsite = nrow(grt.train.inputs$d1), # number of sites
        y = trainData$detHists, # labels in detection component
        rel = trainData$index # correspondence between rows of two component
      )
      # For training observation predictions ==================================
      result.train = grt.predict(model, grt.train.inputs, predict_func,
                                 func.param.train, nTrees=nTrees)
      label.train <- matrix(rep(as.integer(trainData$detHists), length(nTrees)),
                            ncol=length(nTrees))
      idx0.train = which(label.train[,1] == 0)
      idx1.train = which(label.train[,1] == 1)
      # For training prob. predictions
      occ.true.train = trainData$occProbs
      det.true.train = trainData$detProbs
      # Use alternative prediction functions to predict the occupancy/detection
      occPreds.train = grt.predict(model, grt.train.inputs, predict_funcOCC,
                                   func.param.train, nTrees=nTrees)
      detPreds.train = grt.predict(model, grt.train.inputs, predict_funcDET,
                                   func.param.train, nTrees=nTrees)
      # =======================================================================

      # [Validation] ==========================================================
      # Bundle the additional arguments needed to evaluate the model on
      #  the validation data.  Similar to func.param above, except now we
      #  don't need learning parameters like shrinkage and bag.frac.:
      func.param.val <- list(
        nsite = nrow(grt.val.inputs$d1), # number of sites
        y = validateData$detHists, # labels in detection component
        rel = validateData$index # correspondence between rows of two component
      )

      # For validation observation predictions ================================
      result.val = grt.predict(model, grt.val.inputs, predict_func,
                               func.param.val, nTrees=nTrees)
      label.val <- matrix(rep(as.integer(validateData$detHists), length(nTrees)),
                          ncol=length(nTrees))
      idx0.val = which(label.val[,1] == 0)
      idx1.val = which(label.val[,1] == 1)
      # Use alternative prediction functions to predict the occupancy/detection
      occPreds.val = grt.predict(model, grt.val.inputs, predict_funcOCC,
                                 func.param.val, nTrees=nTrees)
      detPreds.val = grt.predict(model, grt.val.inputs, predict_funcDET,
                                 func.param.val, nTrees=nTrees)
      # =======================================================================

      # For the validation loss ===============================================
      # Also calculate loss according to the negative log-likelihood
      bag.flag.val = list(sample1=array(1,c(dim(grt.val.inputs$d1)[1],1)),
                          sample2=array(1,c(dim(grt.val.inputs$d2)[1],1)))

      # Record traiing and validation results for each setting:
      for (t in 1:length(nTrees)) {
        # For train ====================================================
        train.pos.class.score = result.train[idx1.train,t]
        train.neg.class.score = result.train[idx0.train,t]
        # Loss
        train.loss[i,s,t] = model$trainLoss[t]
        # AUROC
        auroc = roc.curve(train.pos.class.score, train.neg.class.score,
                          curve = T)
        train.auroc[i,s,t] = auroc$auc
        # AUPRC
        auprc = pr.curve(train.pos.class.score, train.neg.class.score,
                         curve = T)
        train.auprc[i,s,t] = auprc$auc.davis.goadrich
        if (!is.null(trainData$detProbs)) {
          # For prob. predictions
          occ.true.val = validateData$occProbs
          det.true.val = validateData$detProbs
          # Pearson Correlation
          occ.corr = cor(occ.true.train, occPreds.train[,t], method="pearson")
          train.occCorr[i,s,t] = occ.corr
          det.corr = cor(det.true.train, detPreds.train[,t], method="pearson")
          train.detCorr[i,s,t] = det.corr
        }
        # ==============================================================

        # For validation ===============================================
        val.pos.class.score = result.val[idx1.val,t]
        val.neg.class.score = result.val[idx0.val,t]
        # Loss
        val.loss[i,s,t] = lossFunc(list(occPreds.val[,t],detPreds.val[,t]),
                                   bag.flag.val, func.param.val)[1]
        # AUROC
        auroc = roc.curve(val.pos.class.score, val.neg.class.score, curve = T)
        val.auroc[i,s,t] = auroc$auc
        # AUPRC
        auprc = pr.curve(val.pos.class.score, val.neg.class.score, curve = T)
        val.auprc[i,s,t] = auprc$auc.davis.goadrich
        if (!is.null(trainData$detProbs)) {
          # Pearson Correlation
          occ.corr = cor(occ.true.val, occPreds.val[,t], method="pearson")
          val.occCorr[i,s,t] = occ.corr
          det.corr = cor(det.true.val, detPreds.val[,t], method="pearson")
          val.detCorr[i,s,t] = det.corr
        }
        # ==============================================================
      }

      # 1) Find the best nt based on loss
      opt.t.loss = which.min(val.loss[i,s,])
      occ.pred = occPreds.val[,opt.t.loss] # predicted occupancy probabilitis
      det.pred = detPreds.val[,opt.t.loss] # predicted detection probabilitis
      # 2) Find the best nt based on auprc
      opt.t.auc = which.max(val.auprc[i,s,])
      if (opt.t.loss != opt.t.auc) { # when the opt. nt is different from above
        occ.pred = occPreds.val[,opt.t.auc] # predicted occupancy probabilitis
        det.pred = detPreds.val[,opt.t.auc] # predicted detection probabilitis
      }
    } # s
  } # i

  # For train results =========================================================
  df.train.loss = melt(train.loss)
  colnames(df.train.loss) <- c("depth", "shrinkage", "nTree", "loss")
  df.train.auroc = melt(train.auroc)
  colnames(df.train.auroc) <- c("depth", "shrinkage", "nTree", "auroc")
  df.train.auprc = melt(train.auprc)
  colnames(df.train.auprc) <- c("depth", "shrinkage", "nTree", "auprc")
  df.train.occCorr = melt(train.occCorr)
  colnames(df.train.occCorr) <- c("depth", "shrinkage", "nTree", "occ.corr")
  df.train.detCorr = melt(train.detCorr)
  colnames(df.train.detCorr) <- c("depth", "shrinkage", "nTree", "det.corr")

  df.train.accuracy = merge(df.train.auroc, df.train.auprc,
                            by=c("depth", "shrinkage", "nTree"))
  df.train.corr = merge(df.train.occCorr, df.train.detCorr,
                        by=c("depth", "shrinkage", "nTree"))
  df.train = merge(df.train.loss, df.train.accuracy,
                   by=c("depth", "shrinkage", "nTree"))
  df.train = merge(df.train, df.train.corr,
                   by=c("depth", "shrinkage", "nTree"))
  df.train['bag.frac'] = bagFrac
  df.train['nSites'] = n.site
  df.train['nVisits'] = n.obs.total/n.site
  write.table(df.train, paste(save.path, "df_train.csv",sep=""),
              row.names=F, col.names=T, sep=",")

  # For validation results===================================================
  df.val.loss = melt(val.loss)
  colnames(df.val.loss) <- c("depth", "shrinkage", "nTree", "loss")
  df.val.auroc = melt(val.auroc)
  colnames(df.val.auroc) <- c("depth", "shrinkage", "nTree", "auroc")
  df.val.auprc = melt(val.auprc)
  colnames(df.val.auprc) <- c("depth", "shrinkage", "nTree", "auprc")
  df.val.occCorr = melt(val.occCorr)
  colnames(df.val.occCorr) <- c("depth", "shrinkage", "nTree", "occ.corr")
  df.val.detCorr = melt(val.detCorr)
  colnames(df.val.detCorr) <- c("depth", "shrinkage", "nTree", "det.corr")

  df.val.accuracy = merge(df.val.auroc, df.val.auprc,
                          by=c("depth", "shrinkage", "nTree"))
  df.val.corr = merge(df.val.occCorr, df.val.detCorr,
                      by=c("depth", "shrinkage", "nTree"))
  df.val = merge(df.val.loss, df.val.accuracy,
                 by=c("depth", "shrinkage", "nTree"))
  df.val = merge(df.val, df.val.corr, by=c("depth", "shrinkage", "nTree"))
  df.val['bag.frac'] = bagFrac
  df.val['nSites'] = n.site
  df.val['nVisits'] = n.obs.total/n.site
  write.table(df.val, paste(save.path, "df_val.csv",sep=""),
              row.names=F, col.names=T, sep=",")

  # Set the optimal parameters ==============================================
  if (tuning) {
    best.inter.depth = depths
    best.shrinkage = shrinkages
    best.bag.frac = bagFrac
    bestParamIdxs = which(val.auprc[1,1,]==max(val.auprc[1,1,]),arr.ind=TRUE)
    best.n.iters = nTrees[bestParamIdxs[1]]
  } else {
    bestParamIdxs = which(val.auprc==max(val.auprc),arr.ind=TRUE)
    best.inter.depth = depths[bestParamIdxs[1]]
    best.shrinkage = shrinkages[bestParamIdxs[2]]
    best.n.iters = nTrees[bestParamIdxs[3]]
    best.bag.frac = bagFrac
  }

  # Save the best parameter info.
  best_params = data.frame(best.inter.depth, best.shrinkage, best.n.iters,
                           best.bag.frac)
  colnames(best_params) <- c("best.inter.depth", "best.shrinkage",
                             "best.n.iters", "best.bag.frac")
  write.table(best_params,
              paste(save.path,"best_params.csv",sep=""),
              row.names=FALSE, col.names=FALSE, sep=",")

  return(list(best.inter.depth=best.inter.depth, best.shrinkage=best.shrinkage,
              best.n.iters=best.n.iters))
}

trainOptModel <- function(opts, trainData, grt.train.inputs, save.path,
                          all_trees) {
	# [Training the Optimal model with a training set] ==========================
	# We will use the optimal tc and lr,
  # but we can explore the performance over all nTrees.
	best.inter.depth = opts$best.inter.depth
	best.shrinkage = opts$best.shrinkage

	if (all_trees) {
    # This is just for nt exploration not a actual evalation process
		best.n.iters = n.iters
	} else {
    # If there is no need to track over all nTrees, then use the optimal nt
		best.n.iters = opts$best.n.iters
	}

	n.site <- nrow(trainData$occCovars)
	n.obs.total <- length(trainData$detHists)

	# Bundle additional arguments together as before:
	func.param.trainData <- list(
	  nsite = nrow(grt.train.inputs$d1), # number of sites
	  y = trainData$detHists, # detection histories
	  rel = trainData$index, # correspondence between observations and sites
	  shrinkage = best.shrinkage, # shrinkage for shrinkageStepFunc
	  bag.frac = bagFrac # bagging fraction
	)

	# Set up initial conditions:
	init.f <- list(g1=runif(n.site), g2=runif(n.obs.total))

	# Fit a GRT model with the best tuning parameters to the train data:
	final.model <- grt.fit(
	  nComps=2, # here we have two components: occupancy and detection
	  compInputs=grt.train.inputs, # training data
	  gradFn=gradientFunc, # function calculating the gradient for each component
	  stepLengthFn=shrinkageStepLength, # function calculating step length
	  lossFn=lossFunc, # function calculating value of the objective function
	  sampleFn=samplingFuncSiteFrac, # function for sampling instances
	  otherInputs=func.param.trainData, # other arguments for functions above
	  init=init.f, # initial values for each component function
	  nTrees=best.n.iters, # number of iterations to run (i.e. number of trees)
	  depthFn=best.inter.depth, # interaction depth of trees
	  minObs=1, # minimum number of observations in each leaf of each tree
	  conjGradFlag=TRUE, # whether to use coordinate gradient
	  keepDataFlag=TRUE # whether to keep training data in model
	)

	write.table(final.model$trainLoss, paste(save.path, "train_loss.csv",sep=""),
              row.names=F, col.names=T, sep=",")

	png(paste(save.path,"train_loss.png",sep=""))
	plot(final.model$trainLoss, type="o", col="blue",
       xlab="Iterations", ylab="Loss")
	dev.off()

	return(final.model)
}

testModel <- function(final.model, opts, testData, grt.test.inputs, save.path,
                      all_trees) {
	# [TESTING] =================================================================
	# It's only for evaluating the final model.
	# Still, we can explore the test performance over all nTrees.
	if (all_trees) {
		n.site <- nrow(testData$occCovars)
		n.obs.total <- length(testData$detHists)

		# Bundle the additional arguments needed to evaluate the model on
		#  the test data.  Now we don't need learning parameters
		func.param.test <- list(
			nsite = nrow(grt.test.inputs$d1), # number of sites
			y = testData$detHists, # labels in detection component
			rel = testData$index # correspondence between rows of two component
		)

	  # For test observation predictions ========================================
		result.test = grt.predict(final.model, grt.test.inputs, predict_func,
                              func.param.test, nTrees=nTrees)
		label.test <- matrix(rep(as.integer(testData$detHists), length(nTrees)),
                         ncol=length(nTrees))
		idx0.test = which(label.test[,1] == 0)
		idx1.test = which(label.test[,1] == 1)
		# Use alternative prediction functions to predict the occupancy/detection:
		occPreds.test = grt.predict(final.model, grt.test.inputs, predict_funcOCC,
                                func.param.test, nTrees=nTrees)
		detPreds.test = grt.predict(final.model, grt.test.inputs, predict_funcDET,
                                func.param.test, nTrees=nTrees)
		# =========================================================================

		# For the test loss =======================================================
		# Also calculate loss according to the negative log-likelihood
		bag.flag.test = list(sample1=array(1,c(dim(grt.test.inputs$d1)[1],1)),
												sample2=array(1,c(dim(grt.test.inputs$d2)[1],1)))

		for (t in 1:length(nTrees)) {
			# For test =====================================================
			test.pos.class.score = result.test[idx1.test,t]
			test.neg.class.score = result.test[idx0.test,t]
			# Loss
			test.loss[t] = lossFunc(list(occPreds.test[,t], detPreds.test[,t]),
                              bag.flag.test, func.param.test)[1]
			# AUROC
			auroc = roc.curve(test.pos.class.score, test.neg.class.score, curve = T)
			test.auroc[t] = auroc$auc
			# AUPRC
			auprc = pr.curve(test.pos.class.score, test.neg.class.score, curve = T)
			test.auprc[t] = auprc$auc.davis.goadrich
			if (!is.null(testData$detProbs)) {
				# For test prob. predictions
				occ.true.test = testData$occProbs
				det.true.test = testData$detProbs
				# Pearson Correlation
				occ.corr = cor(occ.true.test, occPreds.test[,t], method="pearson")
				test.occCorr[t] = occ.corr
				det.corr = cor(det.true.test, detPreds.test[,t], method="pearson")
				test.detCorr[t] = det.corr
			}
			# ==============================================================
		}
		df.test.loss = melt(test.loss)
		colnames(df.test.loss) <- c("nTree", "loss")
		df.test.auroc = melt(test.auroc)
		colnames(df.test.auroc) <- c("nTree", "auroc")
		df.test.auprc = melt(test.auprc)
		colnames(df.test.auprc) <- c("nTree", "auprc")
		df.test.occCorr = melt(test.occCorr)
		colnames(df.test.occCorr) <- c("nTree", "occ.corr")
		df.test.detCorr = melt(test.detCorr)
		colnames(df.test.detCorr) <- c("nTree", "det.corr")

		df.test.accuracy = merge(df.test.auroc, df.test.auprc, by=c("nTree"))
		df.test.corr = merge(df.test.occCorr, df.test.detCorr, by=c("nTree"))
		df.test = merge(df.test.loss, df.test.accuracy, by=c("nTree"))
		df.test = merge(df.test, df.test.corr, by=c("nTree"))
		df.test['depth'] = opts$best.inter.depth
		df.test['shrinkage'] = opts$best.shrinkage
		df.test['bag.frac'] = bagFrac
		df.test['nSites'] = n.site
		df.test['nVisits'] = n.obs.total/n.site
		write.table(df.test, paste(save.path, "df_test_", tuning, ".csv",sep=""),
                row.names=F, col.names=T, sep=",")
	} else {
		# Bundle the additional arguments required for the test data
		func.param.test <- list(rel=testData$index)
		result.test <- grt.predict(final.model, grt.test.inputs, predict_func,
                               func.param.test, nTrees=opts$best.n.iters,
                               save.flag=TRUE, save.path)
		occPreds.test = grt.predict(final.model, grt.test.inputs, predict_funcOCC,
                                func.param.test, nTrees=opts$best.n.iters)
		detPreds.test = grt.predict(final.model, grt.test.inputs, predict_funcDET,
                                func.param.test, nTrees=opts$best.n.iters)

		# Loss
		func.param.test <- list(
			nsite = nrow(grt.test.inputs$d1), # number of sites
			y = testData$detHists, # labels in detection component
			rel = testData$index # correspondence between rows of two component
		)
		bag.flag.test = list(sample1=array(1,c(dim(grt.test.inputs$d1)[1],1)),
												sample2=array(1,c(dim(grt.test.inputs$d2)[1],1)))
		test.loss = lossFunc(list(occPreds.test, detPreds.test), bag.flag.test,
                         func.param.test)[1]
		# AUROC
		label.test <- as.integer(testData$detHists)
		idx0.test = which(label.test == 0)
		idx1.test = which(label.test == 1)
		test.pos.class.score = result.test[idx1.test]
		test.neg.class.score = result.test[idx0.test]
		auroc = roc.curve(test.pos.class.score, test.neg.class.score, curve = T)
		test.auroc = auroc$auc
		# AUPRC
		label.test <- as.integer(testData$detHists)
		idx0.test = which(label.test == 0)
		idx1.test = which(label.test == 1)
		test.pos.class.score = result.test[idx1.test]
		test.neg.class.score = result.test[idx0.test]
		auprc = pr.curve(test.pos.class.score, test.neg.class.score, curve = T)
		test.auprc = auprc$auc.davis.goadric
		if (!is.null(testData$detProbs)) {
			# For test prob. predictions
			occ.true.test = testData$occProbs
			det.true.test = testData$detProbs
			# Pearson Correlation
			occ.corr = cor(occ.true.test, occPreds.test, method="pearson")
			test.occCorr = occ.corr
			det.corr = cor(det.true.test, detPreds.test, method="pearson")
			test.detCorr = det.corr
			return (list(loss=test.loss, auroc=test.auroc, auprc=test.auprc,
                   occ.corr=occ.corr, det.corr=det.corr))
		}
		return (list(loss=test.loss, auroc=test.auroc, auprc=test.auprc))
	}
}

FeatureSelection <- function(final.model, iComp) {
	treeSeqs <- final.model$treeSeqs[[iComp]]
	all_fea = c()
	for (i in 1:length(treeSeqs)) {
		fea_list <- treeSeqs[[i]][[1]][[1]]
		tmp <- fea_list[which(fea_list > -1)]
		all_fea = c(all_fea, tmp)
	}
	print(table(all_fea))
	return(table(all_fea))
}