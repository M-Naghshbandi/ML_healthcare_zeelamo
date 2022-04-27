library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(data.table)
library(TTR)
library(forecast)
library(lubridate)
library(randomForest)
library(class)
library(e1071)
library(ROCR)
library(tsfknn)
library(class)
library(car)
library(rpart)
library(MASS)
library(tidyverse)

data <- read_csv("./data/data.csv")
data$date <- as.Date(data$date,format = '%m/%d/%Y')
data$year = lubridate::year(data$date)

######Date Features

data$yday = yday(data$date)
data$quarter = quarter(data$date)
data$month = lubridate::month(data$date)
data$day = lubridate::day(data$date)
data$week = week(data$date)
glimpse(data)
##################
data = as.data.table(data)
data$month = as.factor(data$month)
data$weekdays = factor(data$weekdays,levels = c("Monday", "Tuesday", "Wednesday","Thursday","Friday","Saturday",'Sunday'))
data[weekdays %in% c("Saturday",'Sunday'),weekend:=1]
data[!(weekdays %in% c("Saturday",'Sunday')),weekend:=0]
data$weekend = as.factor(data$weekend)
data$year = as.factor(data$year)
data$quarter = as.factor(data$quarter)
data$week = format(data$date, "%V")
data = as.data.frame(data)
data$week = as.integer(data$week)
glimpse(data)
##############
###############Data Partitioning

N <- nrow(data)
data <- data[sample(1:N),]
#train.set <- data[seq(N/2),]
#test.set <- data[-seq(N/2),]
train.set <- data[1:1354,]
test.set <- data[1355:1403,]

############### Model Evaluation Metrics

mape <- function(real,pred){
    mape <- mean(abs((real - pred)/real))*100
    return (mape)
}
##or
#Acc <- function(pred, real) {
#  acc <- sum(pred == real) / length(pred)
#  return(acc)
#}

################ Linear Regression
#cor(data$issued, data$week)
set.seed(100)
model <- lm(issued ~., train.set)
summary(model)
pred <- predict(model, newdata = test.set)
mape(test.set$issued, pred)
#Acc(test.set$issued,  pred)

#####################Cross Validation on Linear Regression
N <- nrow(data)
data <- data[sample(1:N),]
section <- sample(1:29, size = N,  replace = T)

cross_validation <- function(i){
  train.set <- data[section !=i, ]
  test.set <- data[section == i, ]
  model.lm <- lm(issued ~ ., train.set)
  test.pred.lm <- predict(model.lm, test.set[,-2])
  mape(test.pred.lm, test.set$issued)
  
}
result <- sapply(1:29, cross_validation)

#result <- data.frame(t(result))
#colnames(result) <- c("mape")
#colMeans(result)
#lin.data <- cbind(test.set[,c(2,8)], pred)
#ggplot(lin.data, aes(test.set, pred)) +  geom_point()
#points(lin.data$test.set, lin.data$pred, col = "red", pch=4)
#x <- lin.data[, c(2,8)]
#x.m <- melt(x)
#x <- as.data.frame(lin.data)
#ggplot(x,aes( pred,  issued, color=week)) + geom_point() + geom_smooth(method = "lm")

######################
##Logistic Regression
set.seed(100)
logitMod <- glm(issued ~. , family="gaussian", data = train.set)
summary(logitMod)
predicted <- predict(logitMod, newdata = test.set, type="response") 
mape(test.set$issued, predicted)
#log.data <- cbind(test.set, predicted)
#ggplot(log.data, aes(week, issued)) +  geom_point(aes(color = week))

#####################Cross Validation on Logistic Regression
N <- nrow(data)
data <- data[sample(1:N),]
section <- sample(1:29, size = N,  replace = T)

cross_validation <- function(i){
  train.set <- data[section !=i, ]
  test.set <- data[section == i, ]
  model.glm <- glm(issued ~. , family="gaussian", data = train.set)
  test.pred.glm <- predict(model.glm, test.set[,-2])
  mape(test.pred.glm, test.set$issued)
  
}
result <- sapply(1:29, cross_validation)

##################################
#######Linear Discriminant Analysis
set.seed(100)
model_lda <- lda(issued~.,  train.set)
summary(model_lda)
predictions <- predict(model_lda, newdata = test.set)
mape(test.set$issued, predictions$x)

#####################Cross Validation on Linear Discriminant Analysis
N <- nrow(data)
data <- data[sample(1:N),]
section <- sample(1:29, size = N,  replace = T)

cross_validation <- function(i){
  train.set <- data[section !=i, ]
  test.set <- data[section == i, ]
  model.lda <- lda(issued~.,  train.set)
  test.pred.lda <- predict(model.lda, test.set[,-2])
  mape(test.pred.lda, test.set$issued)
  
}
result <- sapply(1:29, cross_validation)
# Model accuracy
#mean(predictions$class==test.set$issued)
#lda.data <- cbind(test.set, predictions$x)
#ggplot(lda.data, aes(week, issued)) +  geom_point(aes(color = week))

####Classification and Regression Trees
set.seed(100)
 fit <- rpart(issued ~., train.set ,method="class")
 summary(fit)
 predicted <- predict(fit,test.set)
 mape(test.set$issued, predicted)
 cla.data <- cbind(test.set, predicted)
 ggplot(cla.data, aes(week, issued)) +  geom_point(aes(color = week))
 
 # summarize accuracy
# mse <- mean((train.set$issued - predictions)^2)
 #print(mse)
 
 #####################Cross Validation on Classification and Regression Trees
 N <- nrow(data)
 data <- data[sample(1:N),]
 section <- sample(1:29, size = N,  replace = T)
 
 cross_validation <- function(i){
   train.set <- data[section !=i, ]
   test.set <- data[section == i, ]
   model.rpart <- rpart(issued ~., train.set ,method="class")
   test.pred.rpart <- predict(model.rpart, test.set[,-2])
   mape(test.pred.rpart, test.set$issued)
   
 }
 result <- sapply(1:29, cross_validation)
######
##############Naive Bayes Classification
  set.seed(120)  # Setting Seed
 classifier_cl <- naiveBayes(issued ~ ., data = train.set)
summary(classifier_cl)
 pred <- predict(classifier_cl, newdata = test.set)
 mape(test.set$issued, pred)
 #na.data <- cbind(test.set, pred)
 #ggplot(na.data, aes(week, issued)) +  geom_point(aes(color = week))
 
 #####################Cross Validation on Naive Bayes Classification
 N <- nrow(data)
 data <- data[sample(1:N),]
 section <- sample(1:29, size = N,  replace = T)
 
 cross_validation <- function(i){
   train.set <- data[section !=i, ]
   test.set <- data[section == i, ]
   model.naiveBayes <- naiveBayes(issued ~ ., data = train.set)
   test.pred.naiveBayes <- predict(model.naiveBayes, test.set[,-2])
   mape(test.pred.naiveBayes, test.set$issued)
   
 }
 result <- sapply(1:29, cross_validation)
 ######
 
 ################
 #####################################support vector machine
 model <- svm(issued ~ ., train.set)
 test.pred <- predict(model, test.set)
 mape(test.set$issued, test.pred)
 #svm.data <- cbind(test.set, pred)
 #ggplot(svm.data, aes(week, issued)) +  geom_point(aes(color = week))
 #Acc(test.set$issued, test.pred$issued)
 
 #####################Cross Validation on support vector machine
 N <- nrow(data)
 data <- data[sample(1:N),]
 section <- sample(1:29, size = N,  replace = T)
 
 cross_validation <- function(i){
   train.set <- data[section !=i, ]
   test.set <- data[section == i, ]
   model.svm <- svm(issued ~ ., train.set)
   test.pred.svm <- predict(model.svm, test.set[,-2])
   mape(test.pred.svm, test.set$issued)
   
 }
 result <- sapply(1:29, cross_validation)
 ######
 
 ###############
 ###########Random Forest
 set.seed(100)
 rf <- randomForest(issued ~.,  train.set)
 summary(rf)
  pred <- predict(rf, newdata = test.set)
 mape(test.set$issued, pred) 
 #rf.data <- cbind(test.set, pred)
 #ggplot(rf.data, aes(week, issued)) +  geom_point(aes(color = week))
 
 #####################Cross Validation on Random Forest
 N <- nrow(data)
 data <- data[sample(1:N),]
 section <- sample(1:29, size = N,  replace = T)
 
 cross_validation <- function(i){
   train.set <- data[section !=i, ]
   test.set <- data[section == i, ]
   model.rf <- randomForest(issued ~ ., train.set)
   test.pred.rf <- predict(model.rf, test.set[,-2])
   mape(test.pred.rf, test.set$issued)
   
 }
 result <- sapply(1:29, cross_validation)
 ######
 ###########
 ############KKN
 set.seed(100)
 normalize <- function(x) {
   return ((x - min(x)) / (max(x) - min(x))) }
 prc_n <- as.data.frame(lapply(data[,2:8], normalize))
 prc_train <- prc_n[1:1354,]
 prc_test <- prc_n[1355:1402,]
 ran <- sample(1:nrow(data), 0.9 * nrow(data)) 
 
 prc_train_labels <- data[ran,2]
 prc_test_labels <- data[1355:1402,2]  
 prc_test_pred <- knn(train = prc_train, test = prc_test, cl = prc_train_labels, k=1)
 ###########
 
 #####################Cross Validation on kkn
 N <- nrow(data)
 data <- data[sample(1:N),]
 section <- sample(1:29, size = N,  replace = T)
 
 cross_validation <- function(i){
   train.set <- data[section !=i, ]
   test.set <- data[section == i, ]
   model.svm <- svm(issued ~ ., train.set)
   test.pred.svm <- predict(model.svm, test.set[,-2])
   mape(test.pred.svm, test.set$issued)
   
 }
 result <- sapply(1:29, cross_validation)
 ######
 
 
 
 
 