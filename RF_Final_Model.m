%% Final Testing and modelling on Random Forest

clear all
close all

load('Model_RF_parameter.mat');
load('EvaluationMethodParameter.mat','dataValidation');

% load Final Model
RF_model_Final;

% Assess the Final Model Result
[PredRF_Final,ScoreRF_Final]=predict(RF_model_Final,dataValidation);
LossRF_Final=loss(RF_model_Final,dataValidation);
LossRF_Final_Cumulative=loss(RF_model_Final,dataValidation,'mode','cumulative');
[Xrfpr_final,Yrfpr_final,~,AUCRF_Final]=perfcurve(dataValidation.BikeBuyer,ScoreRF_Final(:,2),'1','XCrit',"reca",'YCrit','prec');
F1RF_Final=f1score(dataValidation.BikeBuyer,PredRF_Final);

disp('Error Rate: '+ string(LossRF_Final))
disp('F1 Score: ' + string(F1RF_Final))
disp('AUC - PRC: ' + string(AUCRF_Final))

% Plot against Number of Trees
figure
cla reset
plot(1:800,LossRF_Final_Cumulative,'b-','LineWidth',2)
xlabel('Number of Learning Cycle')
ylabel('Test Error')
title('Final Test Result | Error Rate: ' + string(LossRF_Final*100)+'%')

% Plot Precision Recall Curve
figure
cla reset
x=0:0.1:1;
y=1-x;
plot(Xrfpr_final,Yrfpr_final,'-','Color','b','LineWidth',2)
hold on
plot(x,y,'k:','LineWidth',1)
xlabel('Recall')
ylabel('Precision')
title('Precision-Recall (PR) curves | Random Forest AUC = '+string(AUCRF_Final*100)+'%')
legend({'Random Forest','Random model'})
hold off

% Plot Confusion Chart
figure
confusionchart(dataValidation.BikeBuyer,PredRF_Final,'Title','Random Forest Final Test Matrix | F1 Score: '+ string(F1RF_Final*100)+'%')
