bayesian_func <- function(tc, lr, bf) {
    tc <- round(tc)

    n.iters <- n.iters
    n.site <- nrow(trainDataGlobal$occCovars)
    n.obs.total <- length(trainDataGlobal$detHists)
    init.f <- list(g1=runif(n.site), g2=runif(n.obs.total))

		func.param <- list(
			nsite = nrow(grt.train.inputs.Global$d1), # number of sites
			y = trainDataGlobal$detHists, # detection histories
			rel = trainDataGlobal$index, # correspondence between observations and sites
			shrinkage = lr, # shrinkage for shrinkageStepLength()
			bag.frac = bf # bagging fraction
		)

		model <- grt.fit(
			nComps=2, # here we have two components: occupancy and detection
			compInputs=grt.train.inputs.Global, # training data
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

		func.param.val <- list(
			nsite = nrow(grt.val.inputs.Global$d1), # number of sites
			y = validateDataGlobal$detHists, # labels in detection component
			rel = validateDataGlobal$index # correspondence between rows of two component
		)

		result.val = grt.predict(model, grt.val.inputs.Global, predict_func,
                             func.param.val, nTrees=n.iters)
		label.val <- matrix(rep(as.integer(validateDataGlobal$detHists), 1), ncol=1)
		idx0.val = which(label.val[,1] == 0)
		idx1.val = which(label.val[,1] == 1)

		val.pos.class.score = result.val[idx1.val,1]
		val.neg.class.score = result.val[idx0.val,1]

    auprc = pr.curve(val.pos.class.score, val.neg.class.score, curve = T)
		val.auprc.search = auprc$auc.davis.goadrich

    result <- list(Score = val.auprc.search, Pred = 0)
    return(result)
}