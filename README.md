# StatEcoNet
StatEcoNet: Statistical Ecology Neural Networks for Species Distribution Modeling

---

## Run StatEcoNet/OD-LR/OD-1NN (under StatEcoNet folder)

* The model predicitons (e.g., occupancy, detection, and observation probabilites) will be saved in the "output/prediction" folder.

* The model evaluation metrics (e.g., AUC, Pearson correlation coefficient) will be saved in the "output" folder.

### with synthetic datasets 

Run 2.NN_Testing.ipynb Jupyter notebook

- nSites = [100, 1000]
- nVisits = [3, 10]
- rho = [0, 1]  (0: linear relationships, 1: nonlinear relationships)
- model = [0, 1, 2]  (0: OD-LR, 1: OD-1NN, 2: StatEcoNet)

### with bird species datasets

Run 4.NN_Testing_Birds.ipynb Jupyter notebook

- species = 	[
		"Eurasian Collared-Dove",
                "Common Yellowthroat",
                "Pacific Wren",
                "Song Sparrow",
                "Western Meadowlark"
		]
- test.fold = [1, 2, 3]
- model = [0, 1, 2]  (0: OD-LR, 1: OD-1NN, 2: StatEcoNet)

---

## Run OD-BRT (under OD-BRT folder)

### with synthetic datasets 
```
source("run_syn.R")
runSynthetic(nSites=100,nVisits=3,rho=0)
```
- nSites = 100 or 1000
- nVisits = 3 or 10
- rho = 0 or 1 for linear or nonlinear relationships, respectively

### with bird species datasets
```
source("run_birds.R")
runBirdData("Pacific Wren", test.fold=1)
```
- List of available species names	
	1. Common Yellowthroat
	2. Eurasian Collared-Dove
  3. Pacific Wren
  4. Song Sparrow
  5. Western Meadowlark
- test.fold = 1, 2, or 3  
