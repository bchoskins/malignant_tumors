
#####1: Read in Data
library(tidyverse)
library(caret)
cancer <- read.csv('FNA_cancer.csv')
cancer <- cancer[2:32] #Remove 'id' and 'X' columns
cancer <- na.omit(cancer)
#####2: Perform Basic Exploratory Data Analysis
glimpse(cancer)
# summarize all variables:
summary(cancer)
# correlations
# encode target variable
cor(cancer %>% mutate(diagnosis = ifelse(diagnosis == 'M', 1, 0)))
#Plots to explore diagnoses:
ggplot(cancer, aes(x=diagnosis)) + geom_bar() + ggtitle("Distribution of Diagnosis")
#EDA plots focusing on mean values for now:
ggplot(cancer, aes(radius_mean, diagnosis, fill = diagnosis)) + geom_boxplot() + ggtitle('Radius
Mean by Diagnosis Type')
ggplot(cancer, aes(x=texture_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Texture Mean by Diagnosis Type')
ggplot(cancer, aes(x=perimeter_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Perimeter Mean by Diagnosis Type')
ggplot(cancer, aes(x=area_mean, diagnosis, fill = diagnosis)) + geom_boxplot() + ggtitle('Area
Mean by Diagnosis Type')
ggplot(cancer, aes(x=smoothness_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Smoothness Mean by Diagnosis Type')
ggplot(cancer, aes(x=compactness_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Compactness Mean by Diagnosis Type')
ggplot(cancer, aes(x=concavity_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Concavity Mean by Diagnosis Type')
ggplot(cancer, aes(x=concave.points_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Concave Points Mean by Diagnosis Type')
ggplot(cancer, aes(x=symmetry_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Symmetry Mean by Diagnosis Type')
ggplot(cancer, aes(x=fractal_dimension_mean, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Fractal Dimension Mean by Diagnosis Type')
#Plotted based on work done below to re-look at the most important feature:
ggplot(data=cancer, aes(concave.points_worst, diagnosis, fill = diagnosis)) + geom_boxplot() +
  ggtitle('Worst Concave Points by Diagnosis Type')
ggplot(data = cancer, aes(concave.points_worst, fill = diagnosis)) +
  geom_histogram(binwidth = .01) + ggtitle("Distribution of Worst Concave Points by Diagnosis")
#Checking bivariate relationship between two important features:
ggplot(data = cancer, aes(concave.points_worst, perimeter_worst, color = diagnosis)) +
  geom_point()
#####3: Split Data Into Test and Training Data
n <- nrow(cancer)
set.seed(1842)
test_index <- sample.int(n,size=round(0.2*n))
train <- cancer[-test_index,]
test <- cancer[test_index,]
#Copy for each model to avoid overwriting:
train_decision <- train
test_decision <- test
train_rf <- train
test_rf <- test
train_bg <- train
test_bg <- test
train_knn <- train
test_knn <- test
#####4: Build A Classification Algorithm Using Decision Trees (Pruned)
library(rpart)
library(partykit)
cancer_tree <- rpart(diagnosis~., data=train_decision, cp=0)
plot(as.party(cancer_tree))
#Relative Error Rates:
printcp(cancer_tree)
#plot CP:
plotcp(cancer_tree)
#Prune the tree: (cp from above plot leftmost under line)
new_tree <- prune(cancer_tree, cp=0.029)
plot(as.party(new_tree))
#create a confusion matrix
test_decision$pred <- predict(new_tree, newdata=test_decision,type="class")
confusion_decision <-
  table(test_decision$diagnosis,test_decision$pred,dnn=c("actl","predicted"))
confusion_decision
#Misclassification Rate (error rate):
1-sum(diag(confusion_decision))/sum(confusion_decision)
# caret confusion matrix
confusionMatrix(test_decision$pred , as.factor(test$diagnosis), positive = 'M')
#####5: Build a Classification Algorithm Using Random Forests / Bagging (Adjusted
Parameters)
library(randomForest)
## Random Forest
mtry = sqrt(ncol(train_rf))
cancer_forest <- randomForest(as.factor(diagnosis)~., data = train_rf, mtry=mtry, ntree=400)
cancer_forest
#Importance of vars:
importance(cancer_forest)
#Create the variable importance plot:
varImpPlot(cancer_forest, sort=TRUE)
#Prediction:
forest_pred <- predict(cancer_forest, test_rf, type = "class")
confusion <- table(test_rf$diagnosis, forest_pred)
confusion
#What percent of the values were misclassified:
1-sum(diag(confusion))/length(test_rf$diagnosis)
# caret confusion matrix
confusionMatrix(forest_pred, as.factor(test_rf$diagnosis), positive = 'M')
## Bagging
cancer_bagging <- randomForest(as.factor(diagnosis)~., data = train_bg, mtry=30, ntree=400)
cancer_bagging
#Importance of vars:
importance(cancer_bagging)
#Create the variable importance plot:
varImpPlot(cancer_bagging, sort=TRUE)
#Prediction:
bagging_pred <- predict(cancer_bagging, test_bg, type = "class")
confusion <- table(test_bg$diagnosis, bagging_pred)
confusion
#What percent of the values were misclassified:
1-sum(diag(confusion))/length(test_bg$diagnosis)
# caret confusion matrix
confusionMatrix(bagging_pred, as.factor(test_bg$diagnosis), positive = 'M')
#####6: Build a Classification Algorithm Using Kth Nearest Neighbors (Tuned K)
#Recode the M/B
train_knn$diagnosis <- ifelse(train_knn$diagnosis == "M", 1, 0)
test_knn$diagnosis <- ifelse(test_knn$diagnosis == "M", 1, 0)
#Rescale the data
rescale_x <- function(x) {(x-min(x))/(max(x)-min(x))}
train_knn[2:31] <- rescale_x(train_knn[2:31])
test_knn[2:31] <- rescale_x(test_knn[2:31])
library(class)
# loop through K of 1 through 15
set.seed(1842)
v <- numeric(15)
for (i in 1:15) {
  knn_pred <- knn(train_knn[-1],
                  test=test_knn[-1],
                  cl=train_knn$diagnosis,
                  k=i)
  knn_confusion <- table(knn_pred, test_knn$diagnosis)
  v[i] <- 1 - sum(diag(knn_confusion))/sum(knn_confusion)
}
# plot misclassification rate
plot(v)
#k=1 (Chosen from above for loop finding misclassification rates)
cancer_knn_1 <- knn(train_knn[-1], test=test_knn[-1], cl=train_knn$diagnosis, k=1)
cancer_knn_1
#Confusion Matrix and Misclassification Rate:
conf_1 <- table(cancer_knn_1, test_knn$diagnosis)
1-(sum(diag(conf_1))/nrow(test))
## ROC Curves
library(ROCR)
# decision tree ROC curve
tree_pred_prob <- predict(cancer_tree,newdata = test_decision[1:31], type='prob')
tree_roc_preds <- prediction(tree_pred_prob[,2], test_decision$diagnosis)
tree_roc_perf <- performance(tree_roc_preds, "tpr", "fpr")
# random forest ROC curve
rf_pred_prob <- predict(cancer_forest, newdata = test_rf, type='prob')
rf_roc_preds <- prediction(rf_pred_prob[,2], test_rf$diagnosis)
rf_roc_perf <- performance(rf_roc_preds, "tpr", "fpr")
# bagged trees ROC curve
bg_pred_prob <- predict(cancer_bagging, newdata = test_bg, type='prob')
bg_roc_preds <- prediction(bg_pred_prob[,2], test_bg$diagnosis)
bg_roc_perf <- performance(bg_roc_preds, "tpr", "fpr")
# KNN ROC curve
knn_pred_prob <- knn(train_knn[-1],
                     test=test_knn[-1],
                     cl=train_knn$diagnosis,
                     k=1, prob = T)
knn_preds <- as.vector(knn_pred, mode = "numeric")
test_knn$y <- as.vector(test_knn$diagnosis, mode = "numeric")
knn_roc_preds<-prediction(knn_preds, test_knn$y)
knn_roc_perf <- performance(knn_roc_preds, "tpr", "fpr")
# plot all 4 ROC curves
plot(tree_roc_perf, col='blue')
plot(rf_roc_perf, col='red', add=T)
plot(bg_roc_perf, col='yellow', add=T)
plot(knn_roc_perf, col='green', add=T)
abline(a=0,b=1)
legend( "bottomright", c("decision tree", "random forest", 'bagged trees', "knn"),
        bty = 'n',
        lty=c(1,1,1),
        col=c("blue", "red",'yellow', "green") )
## AUC of each model
tree_auc <- performance(tree_roc_preds, measure = 'auc')
tree_auc@y.values
rf_auc <- performance(rf_roc_preds, measure = 'auc')
rf_auc@y.values
bg_auc <- performance(bg_roc_preds, measure = 'auc')
bg_auc@y.values
knn_auc <- performance(knn_roc_preds, measure = 'auc')
knn_auc@y.values
## Create models using caret library:
# create training controls for caret models
# use 10-fold cross validation
myTrainingControl <- trainControl(method = "cv",
                                  number = 10,
                                  savePredictions = TRUE,
                                  classProbs = TRUE,
                                  verboseIter = TRUE)
# caret tree
treeFit <- train(x = train[-1],
                 y = as.factor(train$diagnosis),
                 method = "rpart",
                 trControl = myTrainingControl,
                 preProcess = c("center","scale"))
# variable importance
varImp(treeFit)
# predict
tree_pred_c <- predict(treeFit,newdata = test)
# make the confusion matrix
confusionMatrix(tree_pred_c, as.factor(test$diagnosis), positive = "M")
# caret RF
randomForestFit <- train(x = train[-1],
                         y = as.factor(train$diagnosis),
                         method = "rf",
                         trControl = myTrainingControl,
                         preProcess = c("center","scale"),
                         ntree = 400)
varImp(randomForestFit)
# predict
rf_pred_c <- predict(randomForestFit,newdata = test)
# make the confusion matrix
confusionMatrix(rf_pred_c, as.factor(test$diagnosis), positive = "M")
# caret bagging
baggedFit <- train(x = train[-1],
                   y = as.factor(train$diagnosis),
                   method = "rf",
                   trControl = myTrainingControl,
                   preProcess = c("center","scale"),
                   ntree = 400, tuneGrid = data.frame(mtry = 30))
varImp(baggedFit)
# predict
bagged_pred_c <- predict(baggedFit,newdata = test)
# make the confusion matrix
confusionMatrix(bagged_pred_c, as.factor(test$diagnosis), positive = "M")
# caret KNN
knnFit <- train(x = train[-1],
                y = as.factor(train$diagnosis),
                method = "knn",
                trControl = myTrainingControl,
                preProcess = c("center","scale"))
varImp(knnFit)
# predict
knn_pred_c <- predict(knnFit,newdata = test)
# make the confusion matrix
confusionMatrix(knn_pred_c, as.factor(test$diagnosis), positive = "M")
