#an exploration of Machine learning Algorithms in the caret package
#on the Poker Hands dataset
#this attempt uses the caret pack to expore the tuning options of the models choosen

#reading the datasets
#read in the poker data sets and store them in two datasets
train <- read.csv("~/PokerHands/poker-hand-training-true.data", header=FALSE)
test <- read.csv("~/PokerHands/poker-hand-testing.data", header=FALSE)

#give the columns names
names(train) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")
names(test) <- c("S1","R1","S2","R2","S3","R3","S4","R4","S5","R5","Hand")

#split the data into 80% training 20% validation
library(caret)
set.seed(12)

trainIndex <- createDataPartition(train$Hand, p = .8,
                                  list = FALSE,
                                  times = 1)
head(trainIndex)

pokertrain <- train[trainIndex,]
pokervalidate <- train[-trainIndex,]

###########################################
# Naive Bayes                             #


#define the resampling method
fitControl <- trainControl(method = "cv",number =10)

#define parameters for tuning
tgrid <- expand.grid(fL = 0 ,usekernel = FALSE)

#separate the attributes from the labels
x <- pokertrain[,1:10]
y <- as.factor(pokertrain$Hand)

fit <- train(x,y, method ="nb",trContronl=fitControl,tuneGrid=tgrid)

pred <- predict(fit, newdata=pokervalidate[,1:10])
confusionMatrix(pred,pokervalidate$Hand)

##########################################
#Boosted Classification Trees            #

#library's need for this investigation
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#define the resampling method
fitControl <- trainControl(method = "none")

#define parameters for tuning
tgrid <- expand.grid(iter = 50, maxdepth = 5 , nu = 1)

#separate the attributes from the labels
x <- pokertrain[,1:10]
y <- as.factor(pokertrain$Hand)


fit <- train(x,y,method ="treebag")#,trContronl=fitControl,tuneGrid=tgrid)

pred <- predict(fit, newdata=test[,1:10])
treeBagCM <- confusionMatrix(pred,test$Hand)
