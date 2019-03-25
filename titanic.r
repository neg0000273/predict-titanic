library(doParallel)
library(randomForest)
library(caret)
library(caretEnsemble)

set.seed(1)

##No fare
train$Fare.Class = rep("",nrow(train))

train$Fare.Class = ifelse(train$Fare == 0,"Free",train$Fare.Class)
train$Fare.Class = ifelse(train$Fare >0 & train$Fare<=50,"Low",train$Fare.Class)
train$Fare.Class = ifelse(train$Fare >50 & train$Fare<=100,"Mid",train$Fare.Class)
train$Fare.Class = ifelse(train$Fare >100 & train$Fare<=200,"High",train$Fare.Class)
train$Fare.Class = ifelse(train$Fare >200,"Very high",train$Fare.Class)
train$Fare.Class = factor(train$Fare.Class)

train$Ticket.Prefix = sapply(train$Ticket, FUN = function(ticket) {
  names <- strsplit(ticket, " ")[[1]]
  ifelse(length(names) > 1, names[1], NA)
})

train$Ticket.Prefix = sub("[.]","",train$Ticket.Prefix)
train$Ticket.Prefix[is.na(train$Ticket.Prefix)] = "None"
train$Ticket.Prefix = factor(train$Ticket.Prefix)

train$Deck = rep("Unknown", nrow(train))

train$Deck = ifelse(grepl("[A]",train$Cabin),"A",train$Deck)
train$Deck = ifelse(grepl("[B]",train$Cabin),"B",train$Deck)
train$Deck = ifelse(grepl("[C]",train$Cabin),"C",train$Deck)
train$Deck = ifelse(grepl("[D]",train$Cabin),"D",train$Deck)
train$Deck = ifelse(grepl("[E]",train$Cabin),"E",train$Deck)
train$Deck = ifelse(grepl("[F]",train$Cabin),"F",train$Deck)
train$Deck = ifelse(grepl("[G]",train$Cabin),"G",train$Deck)

train$Deck = factor(train$Deck)

train$Room.Number = rep(0, nrow(train))
train$Room.Number = unlist(lapply(lapply(lapply(strsplit(train$Cabin," "),sub,pattern="[ABCDEFG]",replacement=""),as.numeric),mean))
train$Room.Number[is.nan(train$Room.Number)] = 1000000
train$Room.Number[is.na(train$Room.Number)] = 1000000

train$Last.Name = unlist(strsplit(train$Name,","))[seq(1,2*nrow(train),2)]
train$Last.Name = factor(train$Last.Name)

##Create a title variable
train$Title <- gsub('(.*, )|(\\..*)', '', train$Name)
train$Title=factor(train$Title)

train$is.French = ifelse(train$Title %in% c("Mme","Mlle"),TRUE,FALSE)
train$is.Clergy = ifelse(train$Title %in% c("Rev"),TRUE,FALSE)
train$is.Noble = ifelse(train$Title %in% c("Sir","the Countess","Jonkheer","Don","Lady"),TRUE,FALSE)
train$is.Military = ifelse(train$Title %in% c("Major","Col"),TRUE,FALSE)
train$is.Child = ifelse(train$Title %in% c("Master"),TRUE,FALSE)
train$is.Doctor = ifelse(train$Title %in% c("Dr"),TRUE,FALSE)
train$is.Married.Woman = ifelse(train$Title %in% c("Mrs", "Mme"),TRUE,FALSE)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

##Predict deck based on fare and family size
newTrain = train[train$Deck != "Unknown",]
newTrain$Deck = droplevels(newTrain$Deck)

x  = model.matrix(Deck~Fare*SibSp*Parch+Fare.Class+is.Noble+Title+Age*Sex+Pclass*Embarked,newTrain)[,-1]
y = newTrain$Deck

##Find variables that have near-zero variance
nearZeroVariance = nearZeroVar(x)
x = x[,-nearZeroVariance]

##Find correlations and remove them
correlations = cor(x)
highCor = findCorrelation(correlations,cutoff = 0.9)
x = x[,-highCor]


train(x,y,data=newTrain,method="rf")


##Preprocess
x  = model.matrix(Survived~Pclass*Embarked+Sex*Age+Fare+SibSp*Parch+Last.Name+Deck*Room.Number+Title+
                    is.French+is.Clergy+is.Noble+is.Military+is.Child+is.Doctor+is.Married.Woman+
                    Fare.Class,train)[,-1]
y = train$Survived

##Find variables that have near-zero variance
nearZeroVariance = nearZeroVar(x)
x = x[,-nearZeroVariance]

##Find correlations and remove them
correlations = cor(x)
highCor = findCorrelation(correlations,cutoff = 0.9)
x = x[,-highCor]

##Training controls
ctrl = trainControl(method = "cv", number = 10,
                    allowParallel = TRUE,classProbs = TRUE, savePredictions="final")
ctrl2 = trainControl(method = "cv", number = 5,
                    classProbs=TRUE,
                    allowParallel = TRUE)

##Grids for caret
##.836
tGridGBM = expand.grid(.n.trees=1500,.interaction.depth=c(12),.shrinkage=c(0.001),
                       .n.minobsinnode=10)

##.82716
tGridRF = expand.grid(.mtry = ncol(x)/3)

tGridLasso = expand.grid(.alpha=1,.lambda=0.012)

#fit = train(x,y,method="glmnet",trControl = ctrl, preProc = c("center","scale"),tuneGrid = tGridLasso)

  modelList = caretList(x, y, trControl = ctrl,
    tuneList = list(caretModelSpec(method="gbm",tuneGrid=tGridGBM,preProc=c("center","scale")),
                    caretModelSpec(method = "glmnet",tuneGrid=tGridLasso, preProc=c("center","scale")),
                    caretModelSpec(method = "rf",tuneGrid=tGridRF, preProc=c("center","scale"))))
     
  modelCor(resamples(modelList))
  
  grEnsemble = caretEnsemble(modelList, trControl = ctrl2)
  summary(grEnsemble)

stopCluster(cl)