# DATA-SCI-B
HackBio internship task# group for paricipants interested in data science. We will provide step by step description on how to perform Machine Learning (ML) utilizing R and Galaxy. 


### Contributors  
* @Mamanu worked on the summary of the tutorial and created the GitHub repo  
* @Anne worked on the analysis of the unsupervised learning  
* @Toyincom worked on the summary of the tutorial  
* @dele_tunde worked on the analysis of the unsupervised learning and in creating the markdown  
* @Chinaza worked on the supervised learning analysis, combining the markdown and poster creation

#### Procedure  

##### Install required packages  

* Installing required packages
```{r}
#install.packages("tidyverse")  
#install.packages("GGally")    
#install.packages("caret")  
#install.packages("gmodels")    
#install.packages("rpart")  
#install.packages("rpart.plot")    
#install.packages("dendextend")  
#install.packages("randomForest")   
#install.packages("mlr3")  
#install.packages("devtools")  
```  
* Install Bioconductor packages  

```{r}
if (!requireNamespace("BiocManager", quietly = TRUE)) 
install.packages("BiocManager") 
BiocManager::install()
BiocManager::install(c("limma", "edgeR"))  
```

* Install libraries from GitHub source  

```{r}
library(devtools)
install_github("vqv/ggbiplot")
library(tidyverse)
```

#### Loading and exploring data  

* Loading the [breast cancer] (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)  from the UCI Machine Learning Repository  


```{r}
library(tidyverse)
breastCancerData <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", col_names=FALSE)
breastCancerDataColNames <- read_csv("https://raw.githubusercontent.com/fpsom/2020-07-machine-learning-sib/master/data/wdbc.colnames.csv", col_names = FALSE)
colnames(breastCancerData) <- breastCancerDataColNames$X1
#* Check out the first 6 lines of the dataset and make the diagnosis column a factor  
breastCancerData %>% head()  
breastCancerData$Diagnosis <- as.factor(breastCancerData$Diagnosis)  
```
