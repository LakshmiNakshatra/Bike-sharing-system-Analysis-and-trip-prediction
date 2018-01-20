# Importing the dataset
bike=read.csv("Bike_data.csv",header=TRUE)

# Attaching dataset to the path
attach(bike)

# Encoding categorical independent variables as Factors
bike$season=factor(bike$season, levels=c("1","2","3","4"), labels=c("spring","summer","fall","winter"))
bike$year=factor(bike$year, levels=c("0","1"), labels=c("2011", "2012"))
bike$month=factor(bike$month, levels=c("1","2","3","4","5","6","7","8","9","10","11","12"), labels=c("Jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
bike$holiday=factor(bike$holiday, levels=c(1,0), labels=c("holiday","not"))
bike$weekday=factor(bike$weekday, levels=c(0,1,2,3,4,5,6), labels=c("Sun","Mon","Tue","Wed","Thu","Fri","Sat"))
bike$weathersit=factor(bike$weathersit, levels=c(1,2,3,4), labels=c("Clear","Mist","Light Snow","Heavy Rain"))

# Encoding categorical dependent variable as Factor
count01=rep("Low",711)
count01[bike$count>median(bike$count)]="High"

# Merging predictors and response
df.bike=data.frame(bike[,-11],count01)

# Splitting immported dataset into train and test sets
set.seed(1)
train <- sample(711,600)
train.set=df.bike[train,]
test.set=df.bike[-train,]
count01.test=count01[-train]

## Fitting Logistic regression
set.seed(1)
logistic.fit=glm(count01~atemp+weekday+month+windspeed+year+hum, data=df.bike, subset=train, family="binomial")
summary(logistic.fit)
# Predicting the results
logistic.prob=predict(logistic.fit,test.set,type="response")
logistic.pred=rep("Low",111)
logistic.pred[logistic.prob>0.075]="High"
table(logistic.pred,count01.test)

## Fitting Tree model
library(tree)
tree.bike = tree(count01~.,df.bike,subset=train)
tree.pred=predict(tree.bike, test.set, type="class")
table(tree.pred, count01.test)
# Cross-validatoin to find the optimal number of trees
set.seed(1)
cv.bike=cv.tree(tree.bike,FUN=prune.misclass)
plot(cv.bike$size,cv.bike$dev,type="b")
# Tree pruning
prune.bike1=prune.misclass(tree.bike,best=7)
prune.pred=predict(prune.bike1,test.set,type="class")
table(prune.pred,count01.test)
summary(prune.bike1)

# Linear Discriminant Analysis(LDA)
library(MASS)
lda.fit=lda(count01~atemp+weekday+month+windspeed+year+hum, data=df.bike,subset=train)

lda.pred=predict(lda.fit,test.set)
table(lda.pred$class, count01.test)

# Quadratic Discriminant Analysis(QDA)
#library(MASS)
qda.fit=qda(count01~atemp+weekday+month+windspeed+year+hum, data=df.bike,subset=train)

qda.pred=predict(qda.fit,test.set)
table(qda.pred$class, count01.test)

# k-Nearest Neighbours(KNN)
library(class)
train.x=cbind(atemp,weekday,month,windspeed,year,hum)[train,]
test.x=cbind(atemp,weekday,month,windspeed,year,hum)[-train,]
train.y=count01[train]

set.seed(1)
knn.pred=knn(train.x,test.x,train.y,k=5)
table(knn.pred,count01.test)

# Maximum Margin Classifier
library(e1071)
set.seed(1)
mm.fit=svm(count01~atemp+weekday+month+windspeed+year+hum,data=df.bike,subset=train,kernel="linear")

mm.pred=predict(mm.fit,test.set)
table(mm.pred,count01.test)

## Support Vector Classifier(Linear Kernel)
library(e1071)
# Parameter tuning
set.seed(1)
svc.tune=tune(svm,count01~atemp+weekday+month+windspeed+year+hum,data=train.set,kernel="linear",
              ranges=list(cost=c(0.01,0.1,1,5,10,100,1000)))
summary(svc.tune)
# Fitting model and predicting results
svc.fit=svm(count01~atemp+weekday+month+windspeed+year+hum,data=df.bike,subset=train,kernel="linear",cost=5)
test.pred=predict(svc.fit,test.set)
table(test.pred,count01.test)

## Support Vector Classifier(Polynomial Kernel)
# Parameter tuning
svmpoly.tune=tune(svm,count01~atemp+weekday+month+windspeed+year+hum,data=train.set,kernel="polynomial",
                  ranges=list(cost=c(0.01,0.1,1,5,10,100,1000), degree=(1:5)))
svmpoly.tune$best.parameters
# Fitting model and predicting results
svmpoly.fit=svm(count01~atemp+weekday+month+windspeed+year+hum,data=df.bike,subset=train,kernel="polynomial",
                cost=1000, degree=2)
poly.pred=predict(svmpoly.fit,test.set)
table(poly.pred,count01.test)

# Support Vector Classifier(Radial Kernel)
# Parameter tuning
svmrad.tune=tune(svm,count01~atemp+weekday+month+windspeed+year+hum,data=train.set,kernel="radial",
                  ranges=list(cost=c(0.01,0.1,1,5,10,100,1000),gamma=c(0.001,0.01,0.1,0.5,1,2)))
svmrad.tune$best.parameters
# Fitting model and predicting results
svmrad.fit=svm(count01~atemp+weekday+month+windspeed+year+hum,data=df.bike,subset=train,kernel="radial",
                cost=1000,gamma=0.01)
rad.pred=predict(svmrad.fit,test.set)
table(rad.pred,count01.test)

## Ensemble learning methods
# Bagging
library(randomForest)
set.seed(1)
bag.bike=randomForest(count01~.,data=df.bike,subset=train,mtry=10,ntree=300,importance=TRUE)
bag.pred=predict(bag.bike,newdata=test.set)
table(bag.pred,count01.test)

# Random Forest
set.seed(1)
rf.bike=randomForest(count01~.,data=df.bike,subset=train,mtry=3,ntree=500,importance=TRUE)
rf.pred=predict(rf.bike,newdata=test.set)
table(rf.pred,count01.test)

# Boosting
library(gbm)
set.seed(1)
boost.bike=gbm((unclass(count01)-1)~.,data=df.bike[train,],distribution="bernoulli",n.trees=5000,interaction.depth=4)
boost.pred=predict(boost.bike,newdata=test.set,n.trees=5000)
pred.class=as.factor(ifelse(boost.pred<0.5,"Low","High"))
table(pred.class,count01.test)

