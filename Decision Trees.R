#this will investigate the Decision Tree classifier to optimize it predictor

#library's need for this investigation
library(caret)
library(rattle)
library(rpart.plot)
library(RColorBrewer)


set.seed(12)
#vary cp values by increasing them
mycontrol1 = rpart.control(minsplit=3, minbucket=1, cp=0.001)
mycontrol2 = rpart.control(minsplit=3, minbucket=1, cp=0.002)
mycontrol3 = rpart.control(minsplit=3, minbucket=1, cp=0.004)
mycontrol4 = rpart.control(minsplit=3, minbucket=1, cp=0.008)
mycontrol5 = rpart.control(minsplit=3, minbucket=1, cp=0.016)

tree1 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol1)
tree2 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol2)
tree3 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol3)
tree4 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol4)
tree5 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol5)

par(mfrow=c(2,3))
plotcp(tree1)
plotcp(tree2) 
plotcp(tree3)
plotcp(tree4)
plotcp(tree5)

printcp(tree1)
printcp(tree2)
printcp(tree3)
printcp(tree4)
printcp(tree5)

#only trees 1,2 and three are viable
#select the complexity parameter associated with minimum error
tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]
tree2$cptable[which.min(tree2$cptable[,"xerror"]),"CP"]
tree3$cptable[which.min(tree3$cptable[,"xerror"]),"CP"]



#vary minsplit values by increasing them
mycontrol1 = rpart.control(minsplit=1, minbucket=1, cp=0.001)
mycontrol2 = rpart.control(minsplit=2, minbucket=1, cp=0.001)
mycontrol3 = rpart.control(minsplit=3, minbucket=1, cp=0.001)
mycontrol4 = rpart.control(minsplit=4, minbucket=1, cp=0.001)
mycontrol5 = rpart.control(minsplit=5, minbucket=1, cp=0.001)

tree1 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol1)
tree2 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol2)
tree3 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol3)
tree4 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol4)
tree5 = rpart(trainHandsPart~.,method="class", data=trainPart,control = mycontrol5)

par(mfrow=c(2,3))
plotcp(tree1)
plotcp(tree2) 
plotcp(tree3)
plotcp(tree4)
plotcp(tree5)

printcp(tree1)
printcp(tree2)
printcp(tree3)
printcp(tree4)
printcp(tree5)

#only trees 1,2 and three are viable
#select the complexity parameter associated with minimum error
tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]
tree2$cptable[which.min(tree2$cptable[,"xerror"]),"CP"]
tree3$cptable[which.min(tree3$cptable[,"xerror"]),"CP"]
tree4$cptable[which.min(tree4$cptable[,"xerror"]),"CP"]
tree5$cptable[which.min(tree5$cptable[,"xerror"]),"CP"]



summary(tree) # detailed summary of splits
#plot(tree)
#text(tree)

#this is a large tree can it be pruned?

pruneTree <- prune(tree, cp=0.0022)
plotcp(pruneTree) 
fancyRpartPlot(pruneTree)
#plot(tree)
#text(tree)

#create a predictor based on the pruned Tree and our testing data

pred = predict(pruneTree, newdata=test, type="class")
treeCM = confusionMatrix(pred,testHands)
treeCM