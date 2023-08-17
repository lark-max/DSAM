
<!-- README.md is generated from README.Rmd. Please edit that file -->

# DSAM

<!-- badges: start -->

This package can be used to split rainfall-runoff data into three data
subsets(Train, Test, Validation) with similar data distribution
characteristics. <!-- badges: end -->

## Installation

There are three ways to install this package in Rstudio on your PC:
  
Way 1:  
Download and install directly from CRAN:
``` r
install.packages("DSAM")
```
  
Way 2:  
Firstly ensure that you have installed the package `devtools`:

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
install.packages("DSAM_1.0.0.tar.gz",repos = NULL,type = "source")
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
level of user feedback. The default is FALSE.
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

Chen, J., Zheng F., May R., Guo D., Gupta H., and Maier H. R.(2022).
Improved data splitting methods for data-driven hydrological model
development based on a large number of catchment samples, Journal of
Hydrology, 613.

Zheng, F., Chen J., Maier H. R., and Gupta H.(2022). Achieving Robust
and Transferable Performance for Conservation‐Based Models of Dynamical
Physical Systems, Water Resources Research, 58(5).

Zheng, F., Chen, J., Ma, Y., Chen Q., Maier H. R., and Gupta H.(2023). A
Robust Strategy to Account for Data Sampling Variability in the
Development of Hydrological Models, Water Resources Research, 59(3).
