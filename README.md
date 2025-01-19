
<!-- README.md is generated from README.Rmd. Please edit that file -->

# DSAM

<!-- badges: start -->

This package can be used to split rainfall-runoff data into three data
subsets(Train, Test, Validation) with similar data distribution
characteristics. <!-- badges: end -->

## Installation

There are several ways to install this package:

Way 1:  
Download and install directly from CRAN:

``` r
install.packages("DSAM")
```

Way 2:  
Download and install from github: Firstly ensure that you have installed
the package `devtools`:

``` r
install.packages("devtools")
```

Then you can install the package like so:

``` r
devtools::install_github("lark-max/DSAM")
```

Way 3:  
You can download the zip package provided in the `release` page, and
install the package by hand:

``` r
install.packages("DSAM_1.0.x.tar.gz",repos = NULL,type = "source")
```

After installation, You can then load the package by:

``` r
library(DSAM)
```

## Instruction

The package has several built-in data splitting algorithms, such as
SOMPLEX, MDUPLEX, SBSS.P, etc. These algorithms are encapsulated and
used by the user only by calling the function `dataSplit`.  
Specific call formats such as:

``` r
result = dataSplit(data,list(sel.alg = "SOMPLEX", writeFile = TRUE))
```

The parameters of the function are composed of two parts. The first
parameter `data` is usually the rainfall runoff data in data.frame or
matrix format, and set the first column as the subscript column. For the
specific format requirements, please refer to the documentation of the
built-in dataset(`DSAM_test_smallData`).

The second parameter `control` is a list of customized information, such
as the selected data splitting algorithm, the proportion of subsets,
whether the output file is required and the output file name, etc. These
information have default defaults, you can refer to the `par.default()`
provided in the package, which is shown as below:

``` r
`include.inp`   
Boolean variable that determines whether the input vectors should be included during the Euclidean distance calculation. The default is TRUE.

`seed`  
Random number seed. The default is 1000.

`sel.alg`   
A string variable that represents the available data splitting algorithms including "SOMPLEX", "MDUPLEX", "DUPLEX", "SBSS.P", "SS" and "TIMECON". The default is "MDUPLEX".

`prop.Tr`   
The proportion of data allocated to the training subset, where the default is 0.6.

`prop.Ts`   
The proportion of data allocated to the test subset, where the default is 0.2.

`Train` 
A string variable representing the output file name for the training data subset. The default is "Train.txt".

`Test`  
A string variable representing the output file name for the test data subset. The default is "Test.txt".

`Validation`    
A string variable representing the output file name for the validation data subset. The default is "Valid.txt".

`loc.calib` 
Vector type: When sel.alg = "TIMECON", the program will select a continuous time-series data subset from the original data set, where the start and end positions are determined by this vector, with the first and the second value representing the start and end position in percentage of the original dataset. The default is c(0,0.6), implying that the algorithm selects the first 60% of the data from the original dataset.

`writeFile` 
Boolean variable that determines whether the data subsets need to be output or not. The default is FALSE.

`showTrace` 
Boolean variable that determines the level of user feedback. The default is FALSE.
```

## Example

The following example can be run directly from the user’s Rstudio
client.

``` r
library(DSAM)
## basic example code
data("DSAM_test_smallData")
res.sml = dataSplit(DSAM_test_smallData)

data("DSAM_test_modData")
res.mod = dataSplit(DSAM_test_modData, list(sel.alg = "SBSS.P"))

data("DSAM_test_largeData")
res.lag = dataSplit(DSAM_test_largeData, list(sel.alg = "SOMPLEX"))
```

## Details

The package also integrates the function of adjunct validation. The
metric `AUC` is used to analyze the similarity of data distribution
features among the sample subsets obtained by various data segmentation
algorithms, which can be calculated by invoking the function `getAUC`.  
For example:

``` r
data("DSAM_test_largeData")
res.split = dataSplit(DSAM_test_largeData,list(sel.alg = "TIMECON"))
runoff.calib <- res.split$Calibration$O
runoff.valid <- res.split$Validation$O
res.auc = getAUC(runoff.calib,runoff.valid)
```

In the above example, the value range of `res.auc` is \[0,1\], and the
closer the value is to 0.5, the more similar the data distribution
characteristics between the two sample subsets are.

## References

When using the DSAM package, please refer to the following paper, in which the package was introduced:  
 
Ji Y., Zheng F., Wen J., Li Q., Chen J., Maier H.R. and Gupta H.V. Gupta. (2025) An R package to partition observation data used for model development and evaluation to achieve model generalizability, Environmental Modelling & Software, 183, 106238, [DOI:10.1016/j.envsoft.2024.106238](https://doi.org/10.1016/j.envsoft.2024.106238).  
 
This  package also utilizes information from the following papers:
 
Maier H.R., Zheng F, Gupta H., Chen J., Mai J., Savic D., Loritz R., Wu W., Guo D., Bennett A., Jakeman A., Razavi S., Zhao J. (2023) On how data are partitioned in model development and evaluation: confronting the elephant in the room to enhance model generalization, Environmental Modelling and Software, 167, 105779, [DOI:10.1016/j.envsoft.2023.105779](https://doi.org/10.1016/j.envsoft.2023.105779).
 
Zheng F., Chen J., Ma Y., Chen Q., Maier H.R. and Gupta H. (2023) A robust strategy to account for data sampling variability in the development of hydrological models, Water Resources Research, 59(3), e2022WR033703, [DOI:10.1029/2022WR033703](https://doi.org/10.1029/2022WR033703).
 
Chen J., Zheng F., May R.J., Guo D., Gupta H.V. and Maier H.R. (2022) Improved data splitting methods for data-driven hydrological model development based on a large number of catchment samples, Journal of Hydrology, 613 (Part A), 128340, [DOI:10.1016/j.jhydrol.2022.128340](https://doi.org/10.1016/j.jhydrol.2022.128340).
 
Zheng F., Chen J., Maier H.R. and Gupta H. (2022) Achieving robust and transferable performance for conservation-based models of dynamical physical systems, Water Resources Research, 58(5), 17406751, [DOI:10.1029/2021WR031818](https://doi.org/10.1029/2021WR031818).
 
Guo D., Zheng F., Gupta H.V., Maier H.R. (2020) On the robustness of conceptual rainfall-runoff models to calibration and evaluation dataset splits selection: A large sample investigation, Water Resources Research, 56(3), e2019WR026752, [DOI:10.1029/2019WR026752](https://doi.org/10.1029/2019WR026752).
 
Zheng F., Maier H.R., Wu W., Dandy G.C., Gupta H.V. and Zhang T. (2018) On lack Of robustness In hydrological model development due to absence of guidelines for selecting calibration and evaluation data: Demonstration for data driven models, Water Resources Research, 54(2), 1013-1030, [DOI:10.1002/2017WR021470](https://doi.org/10.1002/2017WR021470).

