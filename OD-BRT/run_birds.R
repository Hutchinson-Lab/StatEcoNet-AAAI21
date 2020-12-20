source("train.R")
library("dplyr")

runBirdData <- function(species.name, test.fold, trial=1) {
  # List of species names
	# 1. Pacific Wren
	# 2. Western Meadowlark
	# 3. Song Sparrow
	# 4. Common Yellowthroat
	# 5. Eurasian Collared-Dove

	# Load simulated datasets
	myData <- loadBirdData(species.name, test.fold)

	if (tuning == FALSE) {
		opt = read.csv("opt/opt_bird_BRT.csv")
		bagFrac <<- opt[opt$speices==species.name & opt$fold==test.fold,
                    'bag.frac']
		shrinkages <<- opt[opt$speices==species.name & opt$fold==test.fold,
                       'shrinkage']
		depths <<- opt[opt$speices==species.name & opt$fold==test.fold,'depth']
    print(paste("opt params: bf=", bagFrac, ", lr=", shrinkages,
                ", tc=", depths, sep=""))
	}

  if (!dir.exists("output")) {dir.create("output")}
  if (!dir.exists("output/predictions")) {dir.create("output/predictions")}
  base.path = "output/predictions/birds/"
  if (!dir.exists(base.path)) {dir.create(base.path)}
  save.path <- paste(base.path, species.name ,"/", sep="")
  if (!dir.exists(save.path)) {dir.create(save.path)}
  save.path <- paste(save.path, "fold", test.fold,"/", sep="")
  if (!dir.exists(save.path)) {dir.create(save.path)}
  save.path <- paste(save.path,"trial", trial,"/", sep="")
  if (!dir.exists(save.path)) {dir.create(save.path)}
	print(paste("save path: ", save.path, sep=""))

  # Train and evalute a BRT model with the loaded dataset.
  outcomes <- trainEvaluteModel(myData, save.path)
	return(outcomes)
}

loadBirdData <- function(species.name, test.fold) {
	if (test.fold > 3) {
		stop("fold number should be less than equal 3.")
	}
	if (test.fold == 1) {
		valid.fold = 3
		train.fold = 2
	} else if (test.fold == 2) {
		valid.fold = 1
		train.fold = 3
	} else {
		valid.fold = 2
		train.fold = 1
	}
  print(paste("train/validation/test folds: ", train.fold, "/", valid.fold,
              "/", test.fold, sep=""))

	trainData = list()
	validateData = list()
	testData = list()
	fullTrain = list()

	dir.path = paste("../data/OR2020/",species.name,"/",sep="")
	print(paste("data path: ", dir.path, sep=""))

  # Load a train set
	comps = readBirdFiles(dir.path, data.type = paste("f", train.fold, sep=""))
	trainData$occCovars  = comps$occCovars
	trainData$detCovars  = comps$detCovars
	trainData$detHists  = comps$detHists
	nSites = nrow(trainData$occCovars)
	nVisits = nrow(trainData$detCovars)/nSites
	index = array(0,c(nSites*nVisits,1))
  idx = 1
  for (i in 1:nSites) {	# for each site i
    for (t in 1:nVisits) {  # for each visit t to site i
      index[idx] = i
      idx = idx + 1
    } # t
  } # i
	trainData$index = index

  # Load a validation set
	comps = readBirdFiles(dir.path, data.type = paste("f", valid.fold, sep=""))
	validateData$occCovars  = comps$occCovars
	validateData$detCovars  = comps$detCovars
	validateData$detHists  = comps$detHists
	nSites = nrow(validateData$occCovars)
	nVisits = nrow(validateData$detCovars)/nSites
	index = array(0,c(nSites*nVisits,1))
  idx = 1
  for (i in 1:nSites) {	# for each site i
    for (t in 1:nVisits) {  # for each visit t to site i
      index[idx] = i
      idx = idx + 1
    } # t
  } # i
	validateData$index = index

  # Load a test set
	comps = readBirdFiles(dir.path, data.type = paste("f", test.fold, sep=""))
	testData$occCovars  = comps$occCovars
	testData$detCovars  = comps$detCovars
	testData$detHists  = comps$detHists
	nSites = nrow(testData$occCovars)
	nVisits = nrow(testData$detCovars)/nSites
	index = array(0,c(nSites*nVisits,1))
  idx = 1
  for (i in 1:nSites) {	# for each site i
    for (t in 1:nVisits) {  # for each visit t to site i
      index[idx] = i
      idx = idx + 1
    } # t
  } # i
	testData$index = index

	return(list(trainData=trainData, validateData=validateData,
              testData=testData, save.path=dir.path))
}

readBirdFiles <- function(dir.path, data.type) { 
	occCovars <- read.csv(paste(dir.path,data.type,"_occCovars.csv",sep=""),
                        header=F)
	detCovars <- read.csv(paste(dir.path,data.type,"_detCovars.csv",sep=""),
                        header=F)
	detHists <- read.csv(paste(dir.path,data.type,"_detHists.csv",sep=""),
                       header=F)
	# ================================================================
	# Combine two featuers to be new detection featuers
	occCovars.k = occCovars %>% slice(rep(1:n(), each = 3))
	detCovars = cbind(occCovars.k, detCovars)
	# ================================================================
	return(list(occCovars=occCovars, detCovars=detCovars, detHists=detHists[,1]))
}