function [means SDs weights regweights errorvalue] = train(complexity, features, trainset, size1, lambda)
%Function that takes in the complexity, lambda and trainset and trains the
%regression model and returns the means, standard deviations, weights,
%regularized weights and the error values of the trainset for the given
%complexity


% Assigning the model complexity
M = complexity;
% the number of columns in the design matrix for the given M (without the
% consideration of phi0(x))
% the features variable represents the number of columns in input matrix, 
% which also includes the target values aprat from the number of features
designcol = int32((features)*(M-1));
designrow = size1;
% Computing means and SDs for the Gaussian distribution function
size = int32(floor(size1/designcol)); 
% For M = 3 this size gives us 66 values for the input data set, so we have
% 92 different values of means and SDs which is what we need.

%Splitting the total matrix into arrays of size values and computing the
%mean and SD for the arrays to get an array of means and SDs

means = zeros(designcol,1);
SDs = zeros(designcol,1);

%This is done to accomodate the fact that phi0(x) is 1 in the design matrix
means(1) = 1;SDs(1) = 1;
temptrainset = trainset(:,2:features+1);
temptrainset = temptrainset(:);
start = int32(1);
for i=2:designcol+1
       length = int32(start+size-1);
       means(i) = mean(temptrainset(start:length));
       SDs(i) = std(temptrainset(start:length));
       start = start+size;
end
clearvars start length i temptrainset size;
% Computing the gaussian basis function for every feature for the given M
% of all the samples
designMatrix = zeros(designrow, designcol+1);
for i=1:designrow
    designMatrix(i,1) = 1;
    designindex = 2;
    for j=1:M-1
        trainindex = 2;
        while trainindex <= features+1
            nr = trainset(i,trainindex)-means(designindex);
            dr = SDs(designindex);
            designMatrix(i,designindex) = exp((-1)*((nr^2)/(2*(dr^2))));
            trainindex = trainindex + 1;
            designindex = designindex+1;
        end
    end
end
clearvars i j trainindex designindex;
transDesign = transpose(designMatrix);
% Computing weights
weights = pinv((transDesign*designMatrix))*(transDesign*trainset(:,1));
% Applying regularization to weights
reg = designcol+1;
regularization = eye(reg).*lambda;
regdesignMatrix = (transDesign*designMatrix)+regularization;
regweights = pinv(regdesignMatrix) *(transDesign*trainset(:,1));
clearvars transDesign;
transerr = designMatrix*weights - trainset(:,1);
% hold the error value of the particular complexity
% can be used to find the ERMS
errorvalue = transpose(transerr) * transerr;
clearvars transerr;
% Use below to plot the errors
%plot(M, errorvalue,'+', 'MarkerSize', 12);