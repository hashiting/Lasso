clear all;
close all;
load('Part2-TrainingData');
Y = Ytrain;
Phi = Xtrain;
lameda = 0.001;
w = Lasso(Y,Phi,lameda);
% w = w(:,1) - w(:,2);
