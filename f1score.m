% Generate F1 Score given the target variable Y and the Predicted Variables

function F1Score = f1score(targetY,PredictY)
    cfm = confusionmat(targetY,PredictY);
    precision = cfm(2,2)/sum(cfm(:,2));
    recall = cfm(2,2)/sum(cfm(2,:));
    F1Score = 2*(precision*recall)/(precision+recall);

end