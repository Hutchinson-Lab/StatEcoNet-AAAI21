source("train.R")

runSynthetic <- function(nSites, nVisits, rho, trial=1) {
  # Load simulated datasets
	myData <- loadSyntheticData(nSites, nVisits, rho)

  # Load optimal parameters
  if (tuning == FALSE) {
		opt = read.csv("opt/opt_syn_BRT.csv")
		bagFrac <<- opt[opt$nSites==nSites & opt$nVisits==nVisits & opt$rho==rho,
                    'bag.frac']
		shrinkages <<- opt[opt$nSites==nSites & opt$nVisits==nVisits &
                       opt$rho==rho, 'shrinkage']
		depths <<- opt[opt$nSites==nSites & opt$nVisits==nVisits & opt$rho==rho,
                   'depth']
		print(paste("opt params: bf=", bagFrac, ", lr=", shrinkages,
                ", tc=", depths, sep=""))
	}

  # Create output folders
  if (!dir.exists("output")) {dir.create("output")}
  if (!dir.exists("output/predictions")) {dir.create("output/predictions")}
  base.path = "output/predictions/synthetic/"
  if (!dir.exists(base.path)) {dir.create(base.path)}
  save.path <- paste(base.path,nSites,"x",nVisits,"x",rho,"/", sep="")
  if (!dir.exists(save.path)) {dir.create(save.path)}
  save.path <- paste(save.path,"trial", trial,"/", sep="")
  if (!dir.exists(save.path)) {dir.create(save.path)}
  print(paste("save path: ", save.path, sep=""))

	# Train and evalute a BRT model with the loaded dataset.
  outcomes <- trainEvaluteModel(myData, save.path)
	return(outcomes)
}

loadSyntheticData <- function(nSites, nVisits, rho) {
	dir.path = paste("../data/Synthetic/", nSites, "x", nVisits, "/rho", rho,
                   "/", sep="")
	print(paste("data path: ", dir.path, sep=""))
	if (!dir.exists(dir.path)) {
		stop("There is no such directory.")
	}

	trainData = get(load(paste(dir.path, "trainData.RData", sep="")))
	validateData = get(load(paste(dir.path, "validData.RData", sep="")))
	testData = get(load(paste(dir.path, "testData.RData", sep="")))

	return(list(trainData=trainData, validateData=validateData,
							testData=testData, save.path=dir.path))
}