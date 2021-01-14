%% Final Testing and modelling on Naive Bayes
clear all
clc
close all
load('Model_NB_parameter.mat')
load('EvaluationMethodParameter.mat','dataValidation');

% load Final Model
NB_final_model;

%% Data Processing

% Age Binning

[sample,~]=size(dataValidation);
AgeGroup=zeros(sample,1);

for s=1:sample
    for i=length(E_Age):-1:2
        if dataValidation.Age(s) < E_Age(i)
           AgeGroup(s)=i-1;       
        end    
    end
end

% Yearly Income binning
for s=1:sample
    for i=length(E_Income):-1:2
        if dataValidation.YearlyIncome(s) < E_Income(i)
           IncomeGroup(s)=i-1;       
        end    
    end
end

% Ave Month Spending binning
for s=1:sample
    for i=length(E_spend):-1:2
        if dataValidation.AveMonthSpend(s) < E_spend(i)
           MonthlySpendGroup(s)=i-1;       
        end    
    end
end

% Scaling Ave Monthly Spending
tbs=dataValidation{:,{'NumberChildrenAtHome','TotalChildren'}};
data_scale = (tbs - data_min)./denominator;

DP_t = array2table([data_scale,AgeGroup,IncomeGroup',MonthlySpendGroup'],'VariableNames',{'NumberChildrenAtHome','TotalChildren','AgeGroup','IncomeGroup','MonthlySpendGrp'});
% Remove origianl predictor
dataValidation= removevars(dataValidation,{'NumberCarsOwned','Age','YearlyIncome','AveMonthSpend','NumberChildrenAtHome','TotalChildren'});
% update test data
dataValidation = [dataValidation(:,1:8),DP_t,dataValidation(:,end)];


%% Assess the Final Model Result
[PredNB_Final,ScoreNB_Final]=predict(NB_final_model,dataValidation);
LossNB_Final=loss(NB_final_model,dataValidation);
[Xnbpr_final,Ynbpr_final,~,AUCnb_Final]=perfcurve(dataValidation.BikeBuyer,ScoreNB_Final(:,2),'1','XCrit',"reca",'YCrit','prec');
F1NB_Final=f1score(dataValidation.BikeBuyer,PredNB_Final);

disp('Error Rate: '+ string(LossNB_Final))
disp('F1 Score: ' + string(F1NB_Final))
disp('AUC - PRC: ' + string(AUCnb_Final))

% Plot Precision Recall Curve
figure
cla reset
x=0:0.1:1;
y=1-x;
plot(Xnbpr_final,Ynbpr_final,'-','Color','b','LineWidth',2)
hold on
plot(x,y,'k:','LineWidth',1)
xlabel('Recall')
ylabel('Precision')
title('Precision-Recall (PR) curves | Naive Bayes AUC = '+string(AUCnb_Final*100)+'%')
legend({'Naive Bayes','Random model'})
hold off

% Plot Confusion Chart
figure
confusionchart(dataValidation.BikeBuyer,PredNB_Final,'Title','Naive Bayes Final Test Matrix | F1 Score: '+ string(F1NB_Final*100)+'%')

