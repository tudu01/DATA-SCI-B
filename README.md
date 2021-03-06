<h1 align="center"> Welcome to Team Data-Sci-B, HackBio 2021 </h1>  

![image](https://user-images.githubusercontent.com/88287648/128190365-2beca13a-9c53-41b0-83cc-185ae3d8a43c.png)

# DATA-SCI-B
HackBio internship task 2 group for paricipants interested in data science. We will provide step by step description on how to perform Machine Learning (ML) utilizing R. 

*Connect with Hackbio*
<a href="https://twitter.com/TheHackbio?s=08" target="blank"><img align="center" src="http://assets.stickpng.com/images/580b57fcd9996e24bc43c53e.png" alt="tbi_internship" height="30" width="30" /></a>
<a href="https://thehackbio.com/" target="blank"><img align="center" src="https://pbs.twimg.com/profile_images/1274819410814029824/dAaLhOpD_400x400.jpg" alt="HackBioWebsite" height="30" width="30" /></a>
<a href="https://www.linkedin.com/company/hackbio" target="blank"><img align="center" src="https://www.freeiconspng.com/thumbs/linkedin-logo-png/linkedin-logo-3.png" alt="linkedin" height="20" width="20" /></a>
</p>



👥 **List of Contributors**
 
Slack User_Name | Contribution |
--------------- | -------------- |
@Chinaza  | Analyzed Supervised-learning data , combined markdown files, designed poster |
@dele_tunde | Analyzed Unsupervised-learning data, created markdown files |
@Mamanu | Drafted tutorial's summary,  Analyzed Unsupervised-learning data, created GitHub repo |
@Anne | Analyzed Unsupervised-learning data |
@Toyincom | Drafted tutorial's summary |

# Introduction
The project explored the application of Machine Learning in Omics studies. We analyzed, interpreted and visualized results obtained from both supervised and unsupervised machine learning techniques on the Breast cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning repository.  

## Workflow

![Image from iOS (1)](https://user-images.githubusercontent.com/58401006/130317834-61b59f03-84ae-46ce-8b58-d6264fbbd9b2.png)

## Procedure  

##### Install required packages  

* Install required packages
```{r}
install.packages("tidyverse")  
install.packages("GGally")    
install.packages("caret")  
install.packages("gmodels")    
install.packages("rpart")  
install.packages("rpart.plot")    
install.packages("dendextend")  
install.packages("randomForest")   
install.packages("mlr3")  
install.packages("devtools")  
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

#### Exploratory Data Analysis  

* Remove the first column  
```{r}
breastCancerDataNoID <- breastCancerData[2:ncol(breastCancerData)]  
```

* View the dataset  
```{r}
breastCancerDataNoID %>% head()
```

* Create a plot of the first 5 variables  
```{r}
library(GGally)
ggpairs(breastCancerDataNoID[1:5], aes(color=Diagnosis, alpha=0.4))
```

![image](https://user-images.githubusercontent.com/58401006/130315956-01c42dd6-d42f-4a45-8cb7-443c254862bf.png)



* Center and scale the data  
```{r}
library(caret)
ppv <- preProcess(breastCancerDataNoID, method=c("center", "scale"))
breastCancerDataNoID_tr <- predict(ppv, breastCancerDataNoID)
```

* Summarize the first 5 column of the original data
```{r}
breastCancerDataNoID[1:5] %>% summary()
```

* Summarize the first 5 columns of the re-centered and scaled data
```{r}
breastCancerDataNoID_tr[1:5] %>% summary()
```

* Check if the plot has changed with the new data  
```{r}
library(GGally)
ggpairs(breastCancerDataNoID_tr[1:5], aes(color=Diagnosis, alpha=0.4))
```
![image (1)](https://user-images.githubusercontent.com/58401006/130315967-e0d92cbd-d25f-4d21-99ea-a64b6baf3f0f.png)



### Unsupervised Learning

#### Dimensionality Reduction and PCA

* Applying PCA to the data

```{r}
ppv_pca <- prcomp(breastCancerData[3:ncol(breastCancerData)], center = TRUE, scale. = TRUE)
```

  + get a summary of the PCA to see the importance of each Principal component; standard deviation, the proportion of variance it captures and the cumulative proportion of variance.

```{r}
summary(ppv_pca)
```

* Deeper look in the PCA object

```{r}
str(ppv_pca)
```

* Information listed from the function above captures:    
  + The center point (**$center**), scaling (**$scale**) and the standard deviation(**$sdev**) of each original variable  
  + The relationship (correlation or anticorrelation, etc) between the initial variables and the principal components (**$rotation**)  
  + The values of each sample in terms of the principal components (**$x**)      

* Visualize result with *ggbiplot library*   

```{r}
library(devtools)
install_github("vqv/ggbiplot")
library(ggbiplot)
ggbiplot(ppv_pca, choices=c(2,3),
         labels=rownames(breastCancerData), 
         ellipse = TRUE,
         groups = breastCancerData$Diagnosis,
         obs.scale = 1,
         var.axes = TRUE, var.scale = 2) + 
  ggtitle("PCA of Breast Cancer Dataset") + 
  theme_minimal() +
  theme(legend.position =  "bottom")
```
![image (2)](https://user-images.githubusercontent.com/58401006/130316005-7fc27c57-8673-45bc-8395-62e150b9684d.png)



#### Clustering

Clustering algorithms group a set of data points into subsets or clusters. 
The algorithms’ goal is to create clusters that are coherent internally, but clearly different from each other externally. 

* Clustering using the algorithm called **k-means**

```{r}
set.seed(1)
km.out <- kmeans(breastCancerData[3:ncol(breastCancerData)], 
                 centers = 2, 
                 nstart = 20)
```

+ Check output

```{r}
str(km.out)
```

Information listed from the function above contains:

  + **$cluster**: a vector of integers (from 1:k) indicating the cluster to which each point is allocated.  
  + **$centers**: a matrix of cluster centers.  
$withinss: vector of within-cluster sum of squares, one component per cluster.  
  + **$tot.withinss**: total within-cluster sum of squares (i.e. sum(withinss)).  
  + **$size**: the number of points in each cluster.  

* Visualize clusters in relationship to the pricipal components computed earlier

```{r}
ggplot(as.data.frame(ppv_pca$x), aes(x=PC1, y=PC2, color=as.factor(km.out$cluster), shape = breastCancerData$Diagnosis)) +
  geom_point( alpha = 0.6, size = 3) +
  theme_minimal()+
  theme(legend.position = "bottom") +
  labs(title = "K-Means clusters against PCA", x = "PC1", y = "PC2", color = "Cluster", shape = "Diagnosis")
```
![image (3)](https://user-images.githubusercontent.com/58401006/130316009-def0bbf7-2f45-46dd-9137-b157cf294a18.png)


* Cross-tabulation- to check how well the cluster of each tumor (clusters 1 and 2) coincide with the labels

```{r}
library(gmodels)
CrossTable(breastCancerData$Diagnosis, km.out$cluster)
```

* Choosing best **K** with the **elbow method**

  + This method uses within-group homogeneity or within-group heterogeneity to evaluate the variability. 
  + Shows percentage of the variance explained by each cluster.

```{r}
kmean_withinss <- function(k) {
  cluster <- kmeans(breastCancerData[3:ncol(breastCancerData)], k)
  return (cluster$tot.withinss)
}
```

* Try for a single **k** e.g 2

```{r}
kmean_withinss(2)
```

* Use the **sapply()** function to run algorithm over a range of **k**

```{r}
# Set maximum cluster
max_k <-20
# Run algorithm over a range of k
wss <- sapply(2:max_k, kmean_withinss)
```

* Save results into a data frame

```{r}
# Create a data frame to plot the graph
elbow <-data.frame(2:max_k, wss)
```

* Visualize **elbow** point with ggplot

```{r}
# Plot the graph with gglop
ggplot(elbow, aes(x = X2.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 20, by = 1))
```
![image (4)](https://user-images.githubusercontent.com/58401006/130316021-9a247666-b0d8-443c-99ec-8331038a9e2c.png)



#### Hierarchical clustering

It is an alternative approach which builds a hierarchy from the bottom-up, and doesn’t require us to specify the number of clusters beforehand but requires extra steps to extract final clusters.

* K-means clustering

  + remove nominal values , retain numerical values
  
```{r}
breastCancerDataScaled <- as.data.frame(scale(breastCancerData[3:ncol(breastCancerData)]))
summary(breastCancerDataScaled)
```

  + create distance matrix

```{r}
dist_mat <- dist(breastCancerDataScaled, method = 'euclidean')
```

  + Perform hierachical clustering

```{r}
hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg)
```
![image (5)](https://user-images.githubusercontent.com/58401006/130316028-b99c7262-1d30-45eb-8d9f-d51c7353bd8a.png)


dendrogram is built and every data point finally merges into a single cluster with the height(distance) shown on the y-axis.

  + Cut the dendrogram in order to create the desired number of clusters

```{r}
cut_avg <- cutree(hclust_avg, k = 2)
plot(hclust_avg, labels = breastCancerData$ID, hang = -1, cex = 0.2,
     main = "Cluster dendrogram (k = 2)", xlab = "Breast Cancer ID", ylab = "Height")
# k: Cut the dendrogram such that exactly k clusters are produced
# border: Vector with border colors for the rectangles. Coild also be a number vector 1:2
# which: A vector selecting the clusters around which a rectangle should be drawn (numbered from left to right)
rect.hclust(hclust_avg , k = 2, border = c("red","green"), which = c(1, 2))
# Draw a line at the height that the cut takes place
abline(h = 18, col = 'red', lwd=3, lty=2)
```
![image (6)](https://user-images.githubusercontent.com/58401006/130316044-bbf6db0c-3222-46af-8685-e6609114f8d6.png)


  + use the **color_branches()** function from the **dendextend** library to visualize our tree with different colored branches.

```{r}
  library(dendextend)
avg_dend_obj <- as.dendrogram(hclust_avg)
# We can use either k (number of clusters), or clusters (and specify the cluster type)
avg_col_dend <- color_branches(avg_dend_obj, k = 2, groupLabels=TRUE)
plot(avg_col_dend, main = "Cluster dendrogram with color per cluster (k = 2)", xlab = "Breast Cancer ID", ylab = "Height")
```
![image (7)](https://user-images.githubusercontent.com/58401006/130316053-57e5a089-c047-405b-9db5-08f89cfa7e7f.png)


  + Change the way the branches are colored, to reflect **Diagnosis** value

```{r}
  avg_col_dend <- color_branches(avg_dend_obj, clusters = breastCancerData$Diagnosis)
plot(avg_col_dend, main = "Cluster dendrogram with Diagnosis color", xlab = "Breast Cancer ID", ylab = "Height")
```
![image (8)](https://user-images.githubusercontent.com/58401006/130316061-ed8f99ce-2ba8-4e26-b982-94479da8967d.png)


  + Clusters in relationship to the Principal components

```{r}
ggplot(as.data.frame(ppv_pca$x), aes(x=PC1, y=PC2, color=as.factor(cut_avg), shape = breastCancerData$Diagnosis)) +
  geom_point( alpha = 0.6, size = 3) +
  theme_minimal()+
  theme(legend.position = "bottom") +
  labs(title = "Hierarchical clustering (cut at k=2) against PCA", x = "PC1", y = "PC2", color = "Cluster", shape = "Diagnosis")
```
![image (9)](https://user-images.githubusercontent.com/58401006/130316079-d666847b-ab93-4820-8f6c-0c839c285920.png)


### Supervised Learning  

#### Decision Trees  

* Create the train and test dataset  
```{r}
set.seed(1000)
ind <- sample(2, nrow(breastCancerData), replace=TRUE, prob=c(0.7, 0.3))
breastCancerData.train <- breastCancerDataNoID[ind==1,]
breastCancerData.test <- breastCancerDataNoID[ind==2,]
```

* Load the library and create our model  
```{r}
library(rpart)
library(rpart.plot)
myFormula <- Diagnosis ~ Radius.Mean + Area.Mean + Texture.SE
breastCancerData.model <- rpart(myFormula, method = "class", data = breastCancerData.train, minsplit = 10, minbucket = 1, maxdepth = 3, cp = -1)
print(breastCancerData.model$cptable)
rpart.plot(breastCancerData.model)
```
![image (10)](https://user-images.githubusercontent.com/58401006/130316088-e65dcc91-3a42-466a-b189-f0ca91832537.png)


* Select the tree with the minimal prediction error  
```{r}
opt <- which.min(breastCancerData.model$cptable[, "xerror"])
cp <- breastCancerData.model$cptable[opt, "CP"]
breastCancerData.pruned.model <- prune(breastCancerData.model, cp = cp)
rpart.plot(breastCancerData.pruned.model)
table(predict(breastCancerData.pruned.model, type="class"), breastCancerData.train$Diagnosis)
```
![image (11)](https://user-images.githubusercontent.com/58401006/130316099-09a2a64c-9248-4fd2-ac75-b4ce2c6f4ae7.png)


* Check how prediction works in dataset
```{r}
BreastCancer_pred <- predict(breastCancerData.pruned.model, newdata = breastCancerData.test, type = "class")
plot(BreastCancer_pred ~ Diagnosis, data = breastCancerData.test, xlab = "Observed", ylab = "Prediction")
table(BreastCancer_pred, breastCancerData.test$Diagnosis)
```
![image (12)](https://user-images.githubusercontent.com/58401006/130316130-405c5857-098e-4f87-9805-f454acd223e3.png)


#### Random Forest  

* Train the model  
```{r}
library(randomForest)
set.seed(1000)
rf <- randomForest(Diagnosis ~ ., data = breastCancerData.train, ntree = 100, proximity = T)
table(predict(rf), breastCancerData.train$Diagnosis)
```

* Investigate the content of the model
```{r}
print(rf)
```

* View the overall performance of the model
```{r}
plot(rf, main = "")
```


![image (20)](https://user-images.githubusercontent.com/58401006/130318091-6f80d5a9-fc5d-4528-bbd1-9304ccf483d4.png)


* Review the variables with the highest importance
```{r}
importance(rf)
varImpPlot(rf)
```
![image (14)](https://user-images.githubusercontent.com/58401006/130316141-5813410b-0625-4e5b-b7f2-49144a683cbe.png)


* Prediction of diagnosis for the test set and select the feature
```{r}
BreastCancer_pred_RD <- predict(rf, newdata = breastCancerData.test)
table(BreastCancer_pred_RD, breastCancerData.test$Diagnosis)
plot(margin(rf, breastCancerData.test$Diagnosis))
result <- rfcv(breastCancerData.train, breastCancerData.train$Diagnosis, cv.fold = 3)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=2))
```
![image (15)](https://user-images.githubusercontent.com/58401006/130316150-be8321a7-47a2-4f26-ac23-ed1e60f00673.png)

![image (16)](https://user-images.githubusercontent.com/58401006/130316154-bd89b674-5e82-4bfa-b1ad-c478293b7cdb.png)


####  Linear Regression

* Find the correlation between variables
```{r}
cor(breastCancerData$Radius.Mean, breastCancerData$Concave.Points.Mean)
cor(breastCancerData$Concave.Points.Mean, breastCancerData$Area.Mean)
```

* Create a short version of the data and build a linear regression model
```{r}
bc <- select(breastCancerData,Radius.Mean,Concave.Points.Mean,Area.Mean)
bc_model_full <- lm(Radius.Mean ~ Concave.Points.Mean + Area.Mean, data = bc)
bc_model_full
```

* Create predictions on training dataset and visualize
```{r}
preds <- predict(bc_model_full)
plot(preds, bc$Radius.Mean, xlab = "Prediction", ylab = "Observed")
abline(a = 0, b = 1)
summary(bc_model_full)
```
![image (17)](https://user-images.githubusercontent.com/58401006/130316173-413fb2a7-562a-4a33-b314-9faba8c2238c.png)

* Split dataset create model and visualize the predictions
```{r}
set.seed(123)
ind <- sample(2, nrow(bc), replace = TRUE, prob = c(0.75, 0.25))
bc_train <- bc[ind==1,]
bc_test <- bc[ind==2,]
(bc_model <- lm(Radius.Mean ~ Concave.Points.Mean + Area.Mean, data = bc_train))
summary(bc_model)
bc_train$pred <- predict(bc_model)
ggplot(bc_train, aes(x = pred, y = Radius.Mean)) + geom_point() + geom_abline(color = "blue")
```
![image (18)](https://user-images.githubusercontent.com/58401006/130316176-0b8db2fa-9f0d-4376-9654-25f8fd9d88a8.png)

* Predict using test data and plot 
```{r}
bc_test$pred <- predict(bc_model, newdata = bc_test)
ggplot(bc_test, aes(x = pred, y = Radius.Mean)) + geom_point() + geom_abline(color = "blue")
```
![image (19)](https://user-images.githubusercontent.com/58401006/130316181-96fdd38b-d491-40a5-a645-7d712844c3d7.png)

* Calculate how much of variability in dependent variable can be explained by the model
```{r}
bc_mean <- mean(bc_train$Radius.Mean)
tss <- sum((bc_train$Radius.Mean - bc_mean) ^2)
err <- bc_train$Radius.Mean-bc_train$pred
rss <- sum(err^2)
(rsq <- 1-(rss/tss))
```


## Reference

1.	Fotis E. Psomopoulos, 2021 Introduction to Machine Learning using R (Galaxy Training Materials). https://training.galaxyproject.org/training-material/topics/statistics/tutorials/intro-to-ml-with-r/tutorial.html Online; accessed Sat Aug 21 2021
2.	Batut et al., 2018 Community-Driven Data Analysis Training for Biology Cell Systems 10.1016/j.cels.2018.05.012
