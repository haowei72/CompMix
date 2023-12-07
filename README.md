# CompMix

## A comprehensive toolkit for environmental mixtures analysis

* **Install all the dependency R packages**
```{r}
install.packages(c("mvtnorm","igraph","glmnet","gglasso","higlasso",
                   "hierNet","caret","e1071",'rJava',
                   'bartMachine', 'SuperLearner', 'gam',
                   'ipred', 'bartMachineJARs', 'car',
                   'missForest', 'itertools', 'iterators',
                   'xgboost', 'bkmr', 'qgcomp', 'gWQS',
                   'matrixStats', 'pROC',"devtools"))
devtools::install_github("umich-cphds/snif")
```

* **Install and load the package**
```{r}
devtools::install_github("haowei72/CompMix")
library(CompMix)
```

* **Overview**

 The users input the dataset consisting of outcome variable y, exposure variables x and covariates z, and specify test.pct to randomly split the dataset into training and testing datasets. By specifying interaction=TRUE, the Comp.Mix automatically calculates all the pairwise interactions among exposure variables. For users who wish to explore the interaction effects among some covariates and exposures, they can do so by including the specific covariates into the exposure variables x. 

* **Simulate data**
  ```{r}
  dat <- lmi_simul_dat(n=1000,p=20,q=5, block_idx=c(1,1,2,2,3,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3), within_rho=0.6,btw_rho=0.1,R2=0.2, effect_size=1,effect_size_i=1, cancel_effect = FALSE)
  ```


* **Example**

The users would like to perform variable selections on main effects of exposures and covariates, and outcome, exposures and covariates are entered. For any individual interactions that the users would like to include in the models, they can add those into the covariate z.
```{r}
res_ex1 <- Comp.Mix(y=dat$y, x=dat$x, z=dat$z, test.pct=0.5, var.select = TRUE, interaction = FALSE, covariates.forcein = FALSE, bkmr.pip=0.5, seed=2023)
```
Results include exposures and covariates that are selected and their coefficients  by Lasso and Elastic-net, as well as sum-squared errors and correlations calculated from the testing data for model comparisons.


  ```
