% Load the CarDataset
CarData = readtable('carcrash_train_new.csv');

% Display the first few rows of the CarDataset
disp(head(CarData));

% Handle missing values
CarData.weight = fillmissing(CarData.weight, 'constant', median(CarData.weight, 'omitnan'));
CarData.ageOFocc = fillmissing(CarData.ageOFocc, 'constant', median(CarData.ageOFocc, 'omitnan'));

% No need to fill missing categorical values with mode, cause there aren't

% Convert categorical variables
CarData.dvcat = categorical(CarData.dvcat);
CarData.airbag = categorical(CarData.airbag);
CarData.seatbelt = categorical(CarData.seatbelt);
CarData.frontal = categorical(CarData.frontal);
CarData.sex = categorical(CarData.sex);
CarData.abcat = categorical(CarData.abcat);
CarData.occRole = categorical(CarData.occRole);
CarData.dead = categorical(CarData.dead);

% Select features and labels
features = CarData(:, {'dvcat', 'weight', 'airbag', 'seatbelt', 'frontal', 'sex', 'ageOFocc', 'yearacc', 'yearVeh', 'abcat', 'occRole', 'deploy'});
labels = CarData.dead;

% Convert table to matrix
X = table2array(varfun(@double, features));
Y = labels;

% Split CarData into training and testing sets
cv = cvpartition(height(CarData), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

% Train a decision tree classifier
treeModel = fitctree(XTrain, YTrain);

% Display the tree
view(treeModel, 'Mode', 'graph');

% Predict on test CarData
YPred = predict(treeModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / length(YTest);
fprintf('Prediction Accuracy: %.2f%%\n', accuracy * 100);
