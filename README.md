# StatEcoNet
StatEcoNet: Statistical Ecology Neural Networks for Species Distribution Modeling

## Run OD-BRT (under OD-BRT folder)

### with synthetic datasets 

- nSites = 100 or 1000
- nVisits = 3 or 10
- rho=0 or 1 for linear or nonlinear relationships, respectively
```
source("run_syn.R")
runSynthetic(nSites=100,nVisits=3,rho=0)
```

### with bird species datasets

- List of available species names	
	1. Common Yellowthroat
	2. Eurasian Collared-Dove
  3. Pacific Wren
  4. Song Sparrow
  5. Western Meadowlark
- test.fold = 1, 2, or 3  
```
source("run_birds.R")
runBirdData("Pacific Wren", test.fold=1)
```
