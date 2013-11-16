% RUN THIS SCRIPT ( It trains the model till the chosen value of M for the
% chosen lambda and other hyperparameters.



% Prediction script that modifies the regression model to make the
% validation error acceptable and test and find the erms for the test set.
% It also finds the predicted target values by running Neural Network

clear all;
close all;
clc;

% Loading all the data required for the training and prediction
load project1_data.mat;

%hold on;

%Change the limit to train upto M = limit ( Suggested M = 16 to obtain the graphs)
limit = M;

errindex = 1;

% variables to store the error values
errorvaluetrain = zeros(1,limit);
errorvaluevalidation = zeros(1,limit);
errorvalueworeg = zeros(1,limit);

for complexity=1:limit
[means SDs weights regweights trainerrorvalue] = train(complexity, features, trainset, size1, lambda);
errorvaluetrain(1,complexity) = trainerrorvalue;
M = complexity;
% Validation
designcol = int32((features)*(M-1));
designrow = size2;

% Computing the gaussian basis function for every feature for the given M
% of all the samples
designMatrix = zeros(designrow, designcol+1);
for i=1:designrow
    designMatrix(i,1) = 1;
    designindex = 2;
    for j=1:M-1
        validateindex = 2;
        while validateindex <= features+1
            nr = validateset(i,validateindex)-means(designindex);
            dr = SDs(designindex);
            designMatrix(i,designindex) = exp((-1)*((nr^2)/(2*(dr^2))));
            validateindex = validateindex + 1;
            designindex = designindex+1;
        end
    end
end
clearvars i j validateindex designindex nr dr;
% Without Applying regularization
%weights = pinv((transDesign*designMatrix))*(transDesign*validateset(:,1));
transerr = designMatrix*weights - validateset(:,1);
errorvalueworeg(1,errindex) = transpose(transerr) * transerr;
% Applying regularization
transerr = designMatrix*regweights - validateset(:,1);
regterm = 0.5*(lambda*transpose(regweights)*regweights);
errorvaluevalidation(1,errindex) = 0.5*(transpose(transerr) * transerr) + regterm;
errindex = errindex + 1;
clearvars transerr;
end
ermsvalidationworeg = zeros(1,limit);
for i=1:limit
    ermsvalidationworeg(1,i) = sqrt((2*errorvalueworeg(1,i))/double(size2));
end
ermsvalidation = zeros(1,limit);
for i=1:limit
     ermsvalidation(1,i) = sqrt((2*errorvaluevalidation(1,i))/double(size2));
end
% Testing
M = 5; %Chosen complexity from regression model
lambda = 14; % Chosen regularization constant from the regression model
%[means SDs weights regweights] = train(M, features, trainset, size1, lambda);
designcol = int32((features)*(M-1));
designrow = data-(size1+size2);
% Computing the gaussian basis function for every feature for the given M
% of all the samples
designMatrix = zeros(designrow, designcol+1);
for i=1:designrow
    designMatrix(i,1) = 1;
    designindex = 2;
    for j=1:M-1
        testindex = 2;
        while testindex <= features+1
            nr = testset(i,testindex)-means(designindex);
            dr = SDs(designindex);
            designMatrix(i,designindex) = exp((-1)*((nr^2)/(2*(dr^2))));
            testindex = testindex + 1;
            designindex = designindex+1;
        end
    end
end
clearvars i j testindex designindex complexity nr dr;
% Applying regularization
prediction = designMatrix*regweights;
transerr = prediction - testset(:,1);
regterm = 0.5*(lambda*transpose(regweights)*regweights);
errorvaluetest = 0.5*(transpose(transerr) * transerr) + regterm;
ermstest = sqrt((2*errorvaluetest)/double((data-(size1+size2))));

% ERMS of training data
ermstrain = zeros(1,limit);
for i=1:limit
     ermstrain(1,i) = sqrt((2*errorvaluetrain(1,i))/double(size1));
end

% Call the neural network toolbox to verify the erms value
tr = nn_model(nninput',nntarget');

close all;

clearvars i regterm errindex trainerrorvalue;

% Output the final answers
rms_lr = ermstest;
rms_nn = sqrt(min(tr.tperf));
fprintf('\nthe model complexity M for the linear regression model is %d\n', M);
fprintf('\nthe regularization parameters lambda for the linear regression model is %f\n', lambda);
fprintf('\nthe root mean square error for the linear regression model is %f\n', rms_lr);
fprintf('\nthe root mean square error for the neural network model is %f\n', rms_nn);

save project1_data.mat;
% Plot the different graphs, suggest to change limit to 16 and uncomment
% the following lines to plot the various graphs

% grid on;
% legend('error values training');
% xlabel('M');
% ylabel('error value');
% title('Error E(w) for training');
% figure;
% hold on;
% grid on;
% plot(1:limit,ermstrain, '+-b');
% xlabel('M');
% ylabel('Training ERMS');
% title('Train ERMS for various complexity');
% figure;
% hold on;
% grid on;
% plot(1:limit,errorvaluevalidation,'+-r');
% xlabel('M');
% ylabel('error value');
% title('Error E(w) of validation for various complexity');
% figure;
% hold on;
% grid on;
% plot(1:limit,ermsvalidationworeg,'bo-');
% plot(1:limit,ermsvalidation,'ro-');
% % plot(M,ermstest, 'go','MarkerSize', 12);
% % %plot(M,ermssvm, 'g*','MarkerSize', 12);
% legend('erms validation without regularization','erms validation after regularization');
% xlabel('M');
% ylabel('erms');
% title('Validation ERMS for various complexity');
% figure;
% hold on;
% grid on;
% plot(1:limit,ermsvalidationworeg,'bo-');
% xlabel('M');
% ylabel('erms');
% title('Validation ERMS before regularization');
% figure;
% hold on;
% grid on;
% plot(1:limit,ermsvalidation,'ro-');
% xlabel('M');
% ylabel('erms');
% title('Validation ERMS after regularization');
% figure;
% scatter(1:length(testset),prediction);
% xlabel('sample number');
% ylabel('predicted target values');
% title('Relevance prediction');
% figure;
% scatter(1:length(testset),testset(:,1));
% xlabel('sample number');
% ylabel('Actual target values');
% title('Ground truth Target values');
% figure;
% plot(transerr);
% xlabel('sample number');
% ylabel('translation from the actual relevancy(target label)');
% title('Accuracy of relevancy prediction');