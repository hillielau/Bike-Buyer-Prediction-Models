clear all
clc
close all

% Load Training data and Validation Method (k-value)
load('EvaluationMethodParameter.mat','k','data','cvp');
%% Naive Bayes BaseLine Model
tic
mdlNB = fitcnb(data,'BikeBuyer',"CVPartition",cvp);
NBtime=toc;

listlossNB = kfoldLoss(mdlNB,'Mode','individual','LossFun','classiferror');
mdlNB_b = fitcnb(data,'BikeBuyer');
trainlossNB=resubLoss(mdlNB_b);
% Plot graph to observe the variance of error within the K-Fold Validation
x_loss=1:k; % Fold no
figure
plot(x_loss,listlossNB,'LineWidth',1,'Color','b')
hold on 
annotation('textbox',[.6 .15 .25 .15],'String','Training Time (s): ' + string(NBtime),'EdgeColor','k')
hold off
title('Naive Bayes Error within ' +string(k) +'-fold Cross Validation')
xlabel('Fold No.')
ylabel('Classification Error')
ylim([0.2,0.3])

lossNB=mean(listlossNB);
disp('NB loss= ' +  string(lossNB) + ' | Variation= ' + string(std(listlossNB)))

clear mdlNB_b
%% ROC curves shows the classification performance for a balanced dataset
[predNB,score_nb] = kfoldPredict(mdlNB);
[Xnbroc,Ynbroc,~,AUCnbroc] = perfcurve(data.BikeBuyer,score_nb(:,2),'1');

figure
cla reset
xcurve=0:1;
plot(Xnbroc,Ynbroc,'-','Color','b','LineWidth',1)
hold on
plot(xcurve,xcurve,'r--','LineWidth',0.6)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Naive Bayes Baseline ROC curves | AUC= ' + string(AUCnbroc))
legend({'8-Fold NB','Random Model'})
hold off

%% Precision-Recall Curves indicate the model is penalized for predicting the majority negative class
% Chart shows RF handles imbalanced dataset better than NB
[Xnbpr,Ynbpr,~,AUCnbpr] = perfcurve(data.BikeBuyer,score_nb(:,2),'1','XCrit',"reca",'YCrit','prec');

ycurve=1-xcurve;
plot(Xnbpr,Ynbpr,'-','Color','b','LineWidth',1)
hold on
plot(xcurve,ycurve,'r--','LineWidth',0.6)
xlabel('Recall')
ylabel('Precision')
title('Naive Bayes Baseline Precision-Recall (PR) curves | AUC= ' + string(AUCnbpr))
legend({'8-Fold NB','Random model'})
hold off
F1_score_NB = f1score(data.BikeBuyer,predNB);

clear AUCnbroc mdlNB Xnbroc Ynbroc  score_nb  predNB bin_i x_loss


%% Feature Selection - Redundant Features
% Redundent predictors since they are highly correlated by locational characteristics
% Categorical Feature: City +StateProvinceName | CountryRegionName are
listF1NB_1=zeros(1,2);
listlossNB_1=zeros(1,2);

%Either [City, StateProvinceName] or CountryRegionName columns may be
%dropped
for i = 1:2 
    if i==1
        col=[1,2];
    else
        col=3;
    end
    mdlNB_1 = fitcnb(data(:,[col, 4:end]),'BikeBuyer',"CVPartition",cvp);
    [predNB_1,~] = kfoldPredict(mdlNB_1);
    listF1NB_1(i)=f1score(data.BikeBuyer,predNB_1);
    listlossNB_1(i) = kfoldLoss(mdlNB_1);
end
[F1_score_NB_1,idx_NB1]=max(listF1NB_1);
[lossNB_1,idx_1]=min(listlossNB_1);

% Decide whether the feature selection improves the model performance
if or(F1_score_NB_1 < F1_score_NB, lossNB_1 > lossNB)
    data_1=data ; 
else
    if idx_NB1==1
        col=[1,2];
    else
        col=3;
    end
    data_1 = data(:,[col,4:15]);
end

%% Model 1 record
tic
mdlNB_1 = fitcnb(data(:,[1:2, 4:end]),'BikeBuyer',"CVPartition",cvp); % Choice with Lower error rate
NBtime_1=toc;
[predNB_1,score_nb_1] = kfoldPredict(mdlNB_1);
mdlNB_1b = fitcnb(data(:,[1:2, 4:end]),'BikeBuyer');
trainlossNB_1=resubLoss(mdlNB_1b);
[Xnbpr_1,Ynbpr_1,~,AUCnbpr_1] = perfcurve(data.BikeBuyer,score_nb_1(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_1=f1score(data.BikeBuyer,predNB_1);
lossNB_1=kfoldLoss(mdlNB_1);

clear col i idx_1 idx_NB1 mdlNB_1 predNB_1  score_nb_1 mdlNB_1b
%% Pearson Correlation Analysis

% Pearson Correlation on numerical features
data_num=data_1{:,9:end-1};

% Principal Component Analysis
data_num_scale= (data_num - min(data_num))./(max(data_num) - min(data_num)); % Scale the data for PCA
corrcoef_data=corr(data_num_scale);
figure
heatmap(corrcoef_data,"XDisplayLabels",data_1.Properties.VariableNames(9:end-1),'YDisplayLabels',data_1.Properties.VariableNames(9:end-1))
title('Pearson Correlation of Input Predictors');
saveas(gcf,'Chart\Correlation_Heapmap.jpg');

clear data_num data_num_scale corrcoef_data

%% NB Model 1b: Test removal of Highly correlated Features
% Number of Children at Home, Total Children, Yearly Income, AveMonthly
% Spend,
corrlist=[10, 11,13,12,14,9]; % sort by correlation in desc. order
listlossNB_2=zeros(1,length(corrlist));
listF1NB_2=zeros(1,length(corrlist));

for i= 1:length(corrlist)
    rng(1);
    col=corrlist(i);
    tic
    mdlNB_2 = fitcnb(data_1(:,[1:col-1,col+1:end]),'BikeBuyer',"CVPartition",cvp);
    NBtime_2 = toc;
    [predNB_2,~] = kfoldPredict(mdlNB_2);
    listlossNB_2(i)=kfoldLoss(mdlNB_2);
    listF1NB_2(i)=f1score(data_1.BikeBuyer,predNB_2);
end

% Compare F1 score and loss
[~,idx]=min(listlossNB_2);
[~,fmax]=max(listF1NB_2);

% Plot Graph showing improvement
figure 
cla reset
yyaxis left
stem([1:length(corrlist)+2],[listlossNB_1,listlossNB_2],'LineWidth',1.5);
hold on
yline(lossNB,'LineStyle',':','LineWidth',0.8);
ylabel('Error Rate')
ylim([0.2 0.26])
hold off

yyaxis right
stem([1:length(corrlist)+2],[listF1NB_1,listF1NB_2],'LineWidth',1.5,'LineStyle',':','Marker','diamond');
title('Performance with the Removal of Predictors')
xlabel('Predictors')
ylabel('F1 Score')
ylim([0.57 0.61])

xlim([0,length(corrlist)+3]);
xticks(1:length(corrlist)+2);
name=[{'City & State','CountryRegionName'},data_1.Properties.VariableNames(corrlist)];
xticklabels(name);
xtickangle(45);
legend({'Valid. Loss','Base Loss','F1 Score'},'location','NW')
saveas(gcf,'Chart\NB_Removal_Predictors.jpg');

% Decide whether the dimension reduction improves the model performance
data_2 = removevars(data_1,'NumberCarsOwned');

% Model 1b Constructing
mdlNB_2 = fitcnb(data_2,'BikeBuyer',"CVPartition",cvp);
[predNB_2,score_nb_2] = kfoldPredict(mdlNB_2);
[Xnbpr_2,Ynbpr_2,~,AUCnbpr_2] = perfcurve(data_2.BikeBuyer,score_nb_2(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_2=f1score(data_2.BikeBuyer,predNB_2);
lossNB_2=kfoldLoss(mdlNB_2);

clear mdlNB_2 predNB_2 score_nb_2 i idx col fmax name  corrlist predNB_1 score_nb_1

%% Age Discretization
%Consider Binning to improve accuracy
% Bin Age into fixed-interval group
edgelist=4:1:20; % interval range
listlossNB_3= zeros(1,length(edgelist)) ; 
listtrainlossNB_3= zeros(1,length(edgelist)) ;
listF1NB_3  = zeros(1,length(edgelist));
tidx = find(string(data_2.Properties.VariableNames) == "Age"); % Column index of Age

dist={'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel','kernel','kernel','kernel','kernel'};

for i = 1:length(edgelist)
    e=edgelist(i);
    edge= 0:e:100; % Assume max age is 100
    Y = discretize(data_2.Age,edge);
    data_3 = [data_2(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'AgeGroup'}),data_2(:,'BikeBuyer')];
    mdlNB_3 = fitcnb(data_3,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
    mdlNB_3b = fitcnb(data_3,'BikeBuyer','DistributionNames',dist);
    listtrainlossNB_3(i)=resubLoss(mdlNB_3b);
    listlossNB_3(i)=kfoldLoss(mdlNB_3);
    [predNB_3,~] = kfoldPredict(mdlNB_3);
    listF1NB_3(i)=f1score(data_2.BikeBuyer,predNB_3);
end
[lmin,lidx]=min(listlossNB_3);
[fmax,fidx]=max(listF1NB_3);
age_interval=edgelist(fidx) ;

% Conclusion: Both loss & f1 score suggest binning into intervals (length=4) improve F1 performance with a slight trade off in total loss

% Construct Model 3
Y = discretize(data_2.Age,0:age_interval:100); %Binning
data_3 = [data_2(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'AgeGroup'}),data_2(:,'BikeBuyer')];
tic
mdlNB_3 = fitcnb(data_3,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
NBtime_3=toc;
mdlNB_3b = fitcnb(data_3,'BikeBuyer','DistributionNames',dist);
lossNB_3=kfoldLoss(mdlNB_3,'Mode','individual');
trainlossNB_3=resubLoss(mdlNB_3b);
[predNB_3,score_nb_3] = kfoldPredict(mdlNB_3);
[Xnbpr_3,Ynbpr_3,~,AUCnbpr_3] = perfcurve(data.BikeBuyer,score_nb_3(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_3=f1score(data_3.BikeBuyer,predNB_3);

clear tidx i  lidx lmin fmax fidx predNB_3 score_nb_3 mdlNB_3 Y e edge mdlNB_3b edgelist
%% NB Model 4: Yearly Income Discretization

% Assume Kernel Distribution 
% Bin YearlyIncome into n bins as there are uncertainties on the max. value
% in final test set

binlist=4:1:20; 
tidx = find(string(data_3.Properties.VariableNames) == "YearlyIncome"); % column index
listlossNB_4 = zeros(1,length(binlist));
listtrainlossNB_4 = zeros(1,length(binlist));
listF1NB_4 = zeros(1,length(binlist));

for i = 1:length(binlist)
    Y = discretize(data_3.YearlyIncome,binlist(i));
    data_4 = [data_3(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'IncomeGroup'}),data_3(:,'BikeBuyer')];
    mdlNB_4 = fitcnb(data_4,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
    mdlNB_4b = fitcnb(data_4,'BikeBuyer','DistributionNames',dist);
    listlossNB_4(i)= kfoldLoss(mdlNB_4);
    listtrainlossNB_4(i)= resubLoss(mdlNB_4b);
    [predNB_4,~] = kfoldPredict(mdlNB_4);
    listF1NB_4(i)=f1score(data_4.BikeBuyer,predNB_4);
end

[lmin,lidx]=min(listlossNB_4);
[fmax,fidx]=max(listF1NB_4);

% Conclusion: 
incomebin=binlist(fidx);
% Max. F1 and reduce error

[Y,E_Income] = discretize(data_3.YearlyIncome,incomebin);
data_4 = [data_3(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'IncomeGroup'}),data_3(:,'BikeBuyer')];
tic
mdlNB_4 = fitcnb(data_4,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
NBtime_4=toc;
mdlNB_4b = fitcnb(data_4,'BikeBuyer','DistributionNames',dist);
lossNB_4=kfoldLoss(mdlNB_4,'Mode','individual');
trainlossNB_4=resubLoss(mdlNB_4b);
[predNB_4,score_nb_4] = kfoldPredict(mdlNB_4);
[Xnbpr_4,Ynbpr_4,~,AUCnbpr_4] = perfcurve(data.BikeBuyer,score_nb_4(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_4=f1score(data_4.BikeBuyer,predNB_4);

clear  lmin lidx fmax fidx predNB_4 mdlNB_4 score_nb_4 tidx Y mdlNB_4b binlist

%% Model 5: Average Monthly Spending Discretization 

% Bin Average Spending Per Month into n bins
binlist=4:1:20;
tidx = find(string(data_4.Properties.VariableNames) == "AveMonthSpend");
listlossNB_5 = zeros(1,length(binlist));
listF1NB_5 = zeros(1,length(binlist));
listtrainlossNB_5 = zeros(1,length(binlist));

for i=1:length(binlist)
    spendbin=binlist(i);
    Y = discretize(data_4.AveMonthSpend,spendbin);
    data_5 = [data_4(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'MonthlySpendGrp'}),data_4(:,'BikeBuyer')];
    mdlNB_5 = fitcnb(data_5,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
    mdlNB_5b = fitcnb(data_5,'BikeBuyer','DistributionNames',dist);
    listlossNB_5(i)= kfoldLoss(mdlNB_5);
    listtrainlossNB_5(i)=resubLoss(mdlNB_5b);
    [predNB_5,~] = kfoldPredict(mdlNB_5);
    listF1NB_5(i)=f1score(data_5.BikeBuyer,predNB_5);
end

[lmin,lidx]=min(listlossNB_5);
[fmax,fidx]=max(listF1NB_5);

%Conclusion:
% The binning of Average Monthly spend improves its F1 Score at high trade
% off of error rate, hence model is not adapted
%%
% For record and comparison
binlist=4:1:20;
tidx = find(string(data_4.Properties.VariableNames) == "AveMonthSpend");
spendbin=binlist(17);
[Y,E_spend] = discretize(data_4.AveMonthSpend,spendbin);
data_5 = [data_4(:,[1:tidx-1,tidx+1:end-1]),table(Y,'VariableNames',{'MonthlySpendGrp'}),data_4(:,'BikeBuyer')];
tic
mdlNB_5 = fitcnb(data_5,'BikeBuyer',"CVPartition",cvp,'DistributionNames',dist);
NBtime_5=toc;
mdlNB_5b = fitcnb(data_5,'BikeBuyer','DistributionNames',dist);
trainlossNB_5=resubLoss(mdlNB_5b);
lossNB_5=kfoldLoss(mdlNB_5,'Mode','individual');
[predNB_5,score_nb_5] = kfoldPredict(mdlNB_5);
[Xnbpr_5,Ynbpr_5,~,AUCnbpr_5] = perfcurve(data.BikeBuyer,score_nb_5(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_5=f1score(data_5.BikeBuyer,predNB_5);

clear binlist  fmax fidx i lidx lmin mdlNB_5 mdlNB_5b predNB_5 score_nb_5  tidx Y

%% Plot Binning Loss 

CategoryBinned={'Age','Income','Monthly Spend'};
loss_catbin=[lossNB_3,lossNB_4,lossNB_5];
trainloss_catbin=[trainlossNB_3 trainlossNB_4 trainlossNB_5];

% Construct error bar
pos=max(loss_catbin)-mean(loss_catbin);
neg=mean(loss_catbin)-min(loss_catbin);

figure;
cla reset
hold on

yyaxis left
errorbar(1:3,mean(loss_catbin),neg,pos,'LineWidth',1,'CapSize',10,'Color','b','Marker','o')
plot(1:3,trainloss_catbin,'m-')
ylabel('Classification Loss')
ylim([0.2 0.26])

yyaxis right
plot(1:3,[F1_score_NB_3,F1_score_NB_4,F1_score_NB_5],'LineWidth',1,'Color','yellow')
ylabel('F1 Score')
set(gca,'XTick',1:3,'XTickLabel',CategoryBinned)
xlim([0 4]);

title('Classification Performance of Binned Predictors')
legend({'Validation loss','Train Loss','F1 Score'},'location','NW')
hold off
saveas(gcf,'Chart\BinnedPerformance.jpg');
clear fidx fmax lmin lidx i mdlNB_5 predNB_5  Y tidx ans spendbin CategoryBinned loss_catbin pos neg 

%% Model 6: Hyperparameter Optimization 

% Scale remaining continous valuable for fitting Kernel bandwidth
data_scale=data_5{:,9:10};
denominator=max(data_scale)-min(data_scale);
data_min=min(data_scale);
t=array2table((data_scale-data_min)./denominator,'VariableNames',data_5.Properties.VariableNames(9:10));
data_6=[data_5(:,1:8),t,data_5(:,11:end)];

clear data_scale  t 

% Hypermeter Optimization by grid search
mdlNB_hyper = fitcnb(data_6,'BikeBuyer','DistributionNames',dist,'OptimizeHyperparameters',{'Kernel','Width'},'HyperparameterOptimizationOptions',struct('CVPartition',cvp,'Optimizer','gridsearch'));

% Confirm Best Hyperparameter
BestWidth=mdlNB_hyper.Width;
BestDist=dist;
BestKernal=mdlNB_hyper.Kernel;
Predictor=mdlNB_hyper.PredictorNames;
classname=[0 1];
%% Construct Model after optimization
tic
mdlNB_6=fitcnb(data_6,'BikeBuyer','CVPartition',cvp,'DistributionNames',dist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname);
NBtime_6=toc;
loss_NB_6=kfoldLoss(mdlNB_6);
mdlNB_6b=fitcnb(data_6,'BikeBuyer','DistributionNames',dist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname);
trainlossNB_6=resubLoss(mdlNB_6b);
[predNB_6,score_nb_6] = kfoldPredict(mdlNB_6);
[Xnbpr_6,Ynbpr_6,~,AUCnbpr_6] = perfcurve(data.BikeBuyer,score_nb_6(:,2),'1','XCrit',"reca",'YCrit','prec');
F1_score_NB_6 = f1score(data.BikeBuyer,predNB_6);

clear predNB_6 score_nb_6 mdlNB_6 mdlNB_6b
%% Tune Class Prior that reflect class distribution
priors = 0.01:0.01:0.99;
listlossNB_7=zeros(k,length(priors));
listF1NB_7=zeros(1,length(priors));

for i = 1:length(priors)
    prior=priors(i);
    mdlNB_7=fitcnb(data_6,'BikeBuyer','CVPartition',cvp,'DistributionNames',dist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname,'Prior',[1-prior, prior]);   
    listlossNB_7(:,i)=kfoldLoss(mdlNB_7,'Mode','individual');
    [predNB_7,~] = kfoldPredict(mdlNB_7);
    listF1NB_7(i) = f1score(data.BikeBuyer,predNB_7);
end

[fmax,fidx]= max(listF1NB_7);
[lmin,lidx] = min(mean(listlossNB_7));
Bestprior = priors(fidx);
loss_NB_7=mean(listlossNB_7);
%% g
% Plot graph on prior vs loss & F1 score
pos=max(listlossNB_7)-mean(listlossNB_7);
neg=mean(listlossNB_7)-min(listlossNB_7);

cla reset
figure(1)
yyaxis left
errorbar(priors,mean(listlossNB_7),neg,pos,'LineWidth',1,'CapSize',10,'Color','b','Marker','o','MarkerFaceColor','r','MarkerSize',4)
ylabel('8-Fold Classification Loss')

yyaxis right
plot(priors,listF1NB_7,'LineWidth',2,'Color','yellow')
ylabel('F1 Score')
title('Classification Performance of Prior Probability Tuning')
xlabel('Prior Dist')
legend({'Loss','F1 Score'},'Location',"best")
saveas(gcf,'Chart\NB_PriorDistPerformance.jpg');

% F1 increase with classification error
% Find Priors where F1 score is higher and loss lower than previous model
goodprior=priors(and(listF1NB_7>=F1_score_NB_6,mean(listlossNB_7)<=loss_NB_6));
std(listlossNB_7(:,goodprior*100));
% Set prior=0.82 (with low variation in k-fold validation) 
Bestprior=0.82;

%% Build up model 7 
tic
mdlNB_7=fitcnb(data_6,'BikeBuyer','CVPartition',cvp,'DistributionNames',dist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname,'Prior',[1-Bestprior, Bestprior]);  
NBtime_7=toc;
mdlNB_7b=fitcnb(data_6,'BikeBuyer','DistributionNames',dist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname,'Prior',[1-Bestprior, Bestprior]);  
% Evaluation
loss_NB_7=kfoldLoss(mdlNB_7,'Mode','individual');
trainlossNB_7=resubLoss(mdlNB_7b);
[predNB_7,score_nb_7] = kfoldPredict(mdlNB_7);
F1_score_NB_7 = f1score(data.BikeBuyer,predNB_7);
[Xnbpr_7,Ynbpr_7,~,AUCnbpr_7] = perfcurve(data.BikeBuyer,score_nb_7(:,2),'1','XCrit',"reca",'YCrit','prec');

clear  predNB_7 score_nb_7 mdlNB_7b  fmax fidx lmin lidx
%% Plot PRC across all Tunings 
figure
cla reset
plot(Xnbpr,Ynbpr,'-','Color','k','LineWidth',0.8)
hold on
plot(Xnbpr_1,Ynbpr_1,'-','Color','b','LineWidth',0.8)
plot(Xnbpr_2,Ynbpr_2,'-','Color','g','LineWidth',0.8)
plot(Xnbpr_3,Ynbpr_3,'-','Color','c','LineWidth',0.8)
plot(Xnbpr_4,Ynbpr_4,'-','Color','c','LineWidth',0.8)
plot(Xnbpr_5,Ynbpr_5,'-','Color','c','LineWidth',0.8)
plot(Xnbpr_6,Ynbpr_6,'-','Color','y','LineWidth',0.8)
plot(Xnbpr_7,Ynbpr_7,'-','Color','m','LineWidth',1.3)
plot(xcurve,ycurve,'r--','LineWidth',0.6) 
xlabel('Recall')
ylabel('Precision')
title('Naive Bayes Precision-Recall (PR) curves | AUC= ' + string(AUCnbpr_7))
legend({'Baseline','(1) Remove Redundant Features','(2) Feature Selection by Correlation','(3) Age Binned (fixed interval)','(4)Yearly Income Binned','(5)Avg. Spending Binned','(6) Kernel & Bandwidth by Grid Search','(7) Class Prior Tuning','Random model'},'Location',"southwest")
hold off
saveas(gcf,'Chart\PRC_NB_final.jpg');

%% Display Tuning Results in Table

AUCnbprcol= [AUCnbpr AUCnbpr_1 AUCnbpr_2 AUCnbpr_3 AUCnbpr_4 AUCnbpr_5 AUCnbpr_6 AUCnbpr_7];
F1nbcol =  [F1_score_NB F1_score_NB_1 F1_score_NB_2 F1_score_NB_3 F1_score_NB_4 F1_score_NB_5 F1_score_NB_6 F1_score_NB_7];
Timenbcol=[NBtime NBtime_1 NBtime_2 NBtime_3 NBtime_4 NBtime_5 NBtime_6 NBtime_7];
klossnbcol=[lossNB lossNB_1 lossNB_2 mean(lossNB_3) mean(lossNB_4) mean(lossNB_5) loss_NB_6 mean(loss_NB_7)];
trainlossnbcol = [trainlossNB,trainlossNB_1,trainloss_NB_2,trainlossNB_3,trainlossNB_4,trainlossNB_5,trainlossNB_6,trainlossNB_7];

resultT=table([1:8]',AUCnbprcol',F1nbcol',klossnbcol',trainlossnbcol',Timenbcol','VariableNames',{'Model_No','PRC_AUC','F1_Score','Cross_Validated_Loss','Training_Loss','Training_Time'});

figure 
cla reset
yyaxis left
plot(resultT.Model_No,resultT.Cross_Validated_Loss,'LineWidth',2,'Color','#084594')
hold on
plot(resultT.Model_No,resultT.Training_Loss,'LineWidth',2,'Color','#41b6c4','LineStyle',':')
ylabel('Error Rate')
ylim([0.19 0.25])
hold off
yyaxis right
plot(resultT.Model_No,resultT.F1_Score,'LineWidth',2,'Color','#e7298a','LineStyle','-')
hold on
plot(resultT.Model_No,resultT.PRC_AUC,'LineWidth',2,'Color','#ef3b2c','LineStyle','-')
hold off
ylim([0.55 0.80])
ylabel('F1 Score/ AUC')
xlim([0 9])
title('Naive Bayes Model Tuning Progress')
legend({'Validation Loss','Training Loss','F1 Score','AUC - Precision Recall Curve'},'location','best')
xticklabels({'Base','(1) Feature Selection)','(2) Feature Selection','(3) Binning)','(4) Binning','(5) Binning','(6) Kernel pdf','(7) Prior'})
xtickangle(20)
saveas(gcf,'Chart\NB_Model_Tuning_Progress.jpg')
%% Final Model of Naive Bayes
data_final=data_6;
% Model Parameter
% K-fold cross validation
k=8; 
cvp; % Cross validation index

Predictor=mdlNB_7.PredictorNames; 
classname=[0 1]; % Binary classification
Bestprior; % Class prior probability
age_interval = 6; % Bin Age into fixed-interval bins with length of 6
E_Age=0:age_interval:100;
E_Income; % Bin edge where Income are binned into 14 bins
% Fit probability distribution type of each predictors
BestDist;  % smoothening applied in categorical predictors
BestWidth; % Kernal bandwidth of numeric predictors
BestKernal; % Kernal type of numeric predictors 
data_min; % Scaling numeric predictor
denominator; %Scaling numeric predictor
E_spend; % Bin edge where spend are binned into 20 bins

%
% Final Model
NB_final_model=fitcnb(data_final,'BikeBuyer','DistributionNames',BestDist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname,'Prior',[1-Bestprior, Bestprior]);
NB_final_CV_model=fitcnb(data_final,'BikeBuyer','CVPartition',cvp,'DistributionNames',BestDist,'Kernel',BestKernal,'Width',BestWidth,'PredictorNames',Predictor,'ClassNames',classname,'Prior',[1-Bestprior, Bestprior]);

save('Model_NB_parameter.mat','NB_final_model','Predictor','classname','Bestprior','age_interval','E_Age','E_Income','BestDist','BestWidth','BestKernal','Bestprior','data_min','denominator','E_spend');
clear neg pos priors x_loss ans
%% Validation Test on Final Model

TrainLoss_Final_NB=resubLoss(NB_final_model);
kLoss_Final_NB=kfoldLoss(NB_final_CV_model);
[pred,score]=kfoldPredict(NB_final_CV_model);

F1_Validation_NB=f1score(data_3.BikeBuyer,pred);
[~,~,~,AUC_Validation_NB]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');

save('NB_Parameter_Assessment\NB_Validation_Result.mat','TrainLoss_Final_NB','kLoss_Final_NB','F1_Validation_NB','AUC_Validation_NB')


clear  klossnbcol neg pos pred score Timenbcol trainloss_catbin 
save('NB_Parameter_Assessment\NB_Model_all_varaible.mat')