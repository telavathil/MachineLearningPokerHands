#an exploration of Machine learning Algorithms in the caret package
#on the Poker Hands dataset
#this attempt uses the caret pack to expore the tuning options of the models choosen
#this attempt will create a training set with equal populations for all classes to resolve the class population imbalance inherant in poker hands, but they will be tested against a sample that refects those imbalances
#library's need for this investigation
library(caret)


#reading the datasets
#read in the poker data sets and store them in two datasets
train <- read.csv("~/PokerHands/poker-hand-training-true.data", header=FALSE)
test <- read.csv("~/PokerHands/poker-hand-testing.data", header=FALSE)

#give the columns names
names(train) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")
names(test) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")

#convert the the Class variable Hand into a factor
train$Hand <- as.factor(train$Hand)
test$Hand <- as.factor(test$Hand)

#create a training set with equal populations for all classes to resolve the class imbalance inherant in poker hands

downSampleTrain <- downSample(train[,1:10], train$Hand, list = FALSE, yname = "Hand")
upSampleTrain <- upSample(train[,1:10], train$Hand, list = FALSE, yname = "Hand")

#separate the attributes from the labels
x <- upSampleTrain[,1:10]
y <- as.factor(upSampleTrain$Hand)


#display difference in training class frequencey
# One figure in row 1 and two figures in row 2
layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))
barchart(table(train$Hand))
barchart(table(downSampleTrain$Hand),main="Hand distribution Down Sampling")
barchart(table(upSampleTrain$Hand),main="Hand distribution after Up Sampling")

###########################################
# Naive Bayes                             #

#library's need for this investigation
library(MASS)
library(klaR)

#define the resampling method
fitControl <- trainControl(method = "cv",number =10)

#define parameters for tuning
tgrid <- expand.grid(fL = 0 ,usekernel = FALSE)

set.seed(12)
fitNB <- train(x,y, method ="nb",trContronl=fitControl,tuneGrid=tgrid)

pred <- predict(fitNB, newdata=test[,1:10] )
CM <- confusionMatrix(pred,test$Hand)

##########################################
#Boosted Classification Trees            #

#library's need for this investigation
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(ipred)
library(plyr)
library(vcd)
library(lattice)


#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(iter = 50, maxdepth = 5 , nu = 1)
tgrid2 <- expand.grid(n.trees = c(10), interaction.depth=c(2) ,shrinkage=c(0.01),n.minobsinnode = c(2) )

#TreeBag model
set.seed(12)
fit <- train(x,y,method ="treebag")
fitTreeBag <- fit
pred <- predict(fit, newdata=test[,1:10])
treeBagCM <- confusionMatrix(pred,test$Hand)

#boosting Tree model
set.seed(12)
fit <- train(x,y,method ="gbm",trContronl=fitControl,tuneGrid=tgrid2)
fitBoost <- fit
pred <- predict(fitBoost, newdata=test[,1:10])
BoostCM <- confusionMatrix(pred,test$Hand)
BoostCM

##############################
# Neural Net                 #


#library's need for this investigation
library(nnet)
library(neuralnet)
library(devtools)
library(caret)



#web reasource to plot neural nets
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


#define the resampling method
fitControl <- trainControl(method = "none")
#define parameters for tuning
tgrid <- expand.grid(size = 4 ,decay = 0.1)

set.seed(12)
fitNN <- train(x,y, method ="nnet",trContronl=fitControl,tuneGrid=tgrid)
plot.nnet(fitNN)

#create a predictor based on the Neural Network and our testing data

pred <- predict(fitNN, newdata=test[,1:10])
neuralNetworkCM = confusionMatrix(pred,test$Hand)
neuralNetworkCM

##############################
#plot the sensitivity and specificity of all the models

barplot(CM$byClass[,1:2],
        beside=T,
        main="Naive Bayes Sensitivity and Specificity of All Class Predictions", 
        #legend(rownames(CM$byClass)),
        ylim=0:1
        )

barplot(treeBagCM$byClass[,1:2],
        beside=T,
        main="Bagged Trees Sensitivity and Specificity of All Class Predictions", 
        #legend(rownames(CM$byClass)),
        ylim=0:1
)

barplot(BoostCM$byClass[,1:2],
        beside=T,
        main="Boosted trees Sensitivity and Specificity of All Class Predictions", 
        #legend(rownames(CM$byClass)),
        ylim=0:1
)

barplot(neuralNetworkCM$byClass[,1:2],
        beside=T,
        main="Neural Network Sensitivity and Specificity of All Class Predictions", 
        #legend(rownames(CM$byClass)),
        ylim=0:1
)