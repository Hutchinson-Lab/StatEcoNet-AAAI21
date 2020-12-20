source("run_syn.R")
source("run_birds.R")

if (!dir.exists("output")) {dir.create("output")}
mulRuns_syn <- function() {
	file.path= paste("output/BRT_syn_test.csv",sep="")

	if (file.exists(file.path)) {
		stop('The file already exists.
          Do you want to add to the existing file? (Yes-1, No-0)')
		if (input != 1) {
			stop("Use a different file name.")
		}
		df.record <- read.csv(file.path)
		i = dim(df.record)[1] + 1
	}
	else {
		df.record <- data.frame(nSites = integer(),
                            nVisits = integer(),
                            rho = integer(),
                            trial = integer(),
                            best.iter = double(),
                            opt.learning.time = double(),
                            test.auroc = double(),
                            test.auprc = double(),
                            test.occ.corr = double(),
                            test.det.corr = double(),
                            stringsAsFactors = FALSE)
		i = 1
	}

	ptm.multi <- proc.time()
	for (nSites in c(100,1000)) {
		for (nVisits in c(3,10)) {
			for (rho in c(0,1)) {
        for (trial in 1:1) {
          cat(c("nSites", nSites, "nVisits", nVisits, "rho", rho,
                "trial", trial, "\n"))
          res = runSynthetic(nSites, nVisits, rho, trial)
          df.record[i,] = c(nSites, nVisits, rho, trial, res$best.nt,
                            res$elapse, res$test$auroc, res$test$auprc,
                            res$test$occ.corr, res$test$det.corr)
          i = i + 1
          write.table(df.record, file.path, row.names=FALSE, col.names=TRUE, sep=",")
        }
			}
		}
	}

	elapsed.multi <- (proc.time() - ptm.multi)[3]
	print(paste("It took ", elapsed.multi, " sec", sep=""))
}

mulRuns_real <- function() {
  species_list = c("Common Yellowthroat",
                   "Eurasian Collared-Dove",
                   "Pacific Wren",
                   "Song Sparrow",
                   "Western Meadowlark")

	filename= paste("output/BRT_bird_test.csv", sep="")

	if (file.exists(filename)) {
		input = readline('The file already exists.
                     Do you want to add to the existing file? (Yes-1, No-0)')
		if (input != 1) {
			stop("Use a different file name.")
		}
		df.record <- read.csv(filename)
		i = dim(df.record)[1] + 1
	}
	else {
		df.record <- data.frame(species.name = character(),
                            test.fold = integer(),
                            trial = integer(),
                            best.iter = double(),
                            opt.learning.time = double(),
                            test.loss = double(),
                            test.auroc = double(),
                            test.auprc = double(),
                            stringsAsFactors = FALSE)
		i = 1
	}

	ptm.multi <- proc.time()
	for (species.name in species_list) {
    for (test.fold in 1:3) {
      for (trial in 1:1) {
        cat(c("species name:", species.name, "\t test_fold:", test.fold,
              "\t trial:", trial, "\n"))
        res = runBirdData(species.name, test.fold, trial)
        df.record[i,] = c(species.name, test.fold, trial, res$best.nt,
                          res$elapse, res$test$loss, res$test$auroc,
                          res$test$auprc)
        i = i + 1
        write.table(df.record, filename, row.names=FALSE, col.names=TRUE, sep=",")
      }
    }
	}
  elapsed.multi <- (proc.time() - ptm.multi)[3]
	print(paste("It took ", elapsed.multi, " sec", sep=""))
}

#mulRuns_syn()
#mulRuns_real()