clear all
clc
close all

% Load Training data and Validation Method (k-value)
load('EvaluationMethodParameter.mat','k','data','cvp');


%% Random Forest Baseline Model
% Assume tree=100
tic
mdlRF = fitensemble(data,'BikeBuyer','Bag',100,'Tree','CVPartition',cvp,'Type','Classification');
RFtime=toc;
listlossRF = kfoldLoss(mdlRF,'Mode','individual','LossFun','classiferror');

% Plot graph to observe the variance of error within the K-Fold Validation
x_loss=1:k; % Fold no
figure
hold on 
plot(x_loss,listlossRF,'LineWidth',1,'Color','g')
annotation('textbox',[.6 .15 .25 .15],'String','Training Time(s):' +string(RFtime),'EdgeColor','k')
hold off
title('Random Forest Error within ' +string(k) +'-fold Cross Validation')
xlabel('Fold No.')
ylabel('Classification Error')
ylim([0.2,0.3])

lossRF=mean(listlossRF);
disp('RF loss= ' +  string(lossRF) + ' | Variation= ' + string(std(listlossRF)))

%% ROC Curve
% ROC curves shows the classification performance for a balanced dataset
[predRF,score_rf] = kfoldPredict(mdlRF);
[Xrfroc,Yrfroc,~,AUCrfroc] = perfcurve(data.BikeBuyer,score_rf(:,2),'1');

figure
cla reset
xcurve=0:1;
plot(Xrfroc,Yrfroc,'-','Color','g','LineWidth',1)
hold on
plot(xcurve,xcurve,'r--','LineWidth',0.6)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Random Forest Baseline ROC curves | AUC: ' +string(AUCrfroc))
legend({'8-Fold RF','Random Model'})
hold off

%% Precision-Recall Curves indicate the model is penalized for predicting the majority negative class
[Xrfpr,Yrfpr,~,AUCrfpr] = perfcurve(data.BikeBuyer,score_rf(:,2),'1','XCrit',"reca",'YCrit','prec');

ycurve=1-xcurve;
plot(Xrfpr,Yrfpr,'-','Color','g','LineWidth',1)
hold on
plot(xcurve,ycurve,'r--','LineWidth',0.6)
xlabel('Recall')
ylabel('Precision')
title('Random Forest Baseline Precision-Recall (PR) curves | AUC= '+ string(AUCrfpr))
legend({'8-Fold RF','Random model'})
hold off

F1_score_RF = f1score(data.BikeBuyer,predRF);

clear AUCrfroc   mdlRF Xrfroc Yrfroc   score_rf  predRF  bin_i x_loss
%% Number of Learning Cycle
clear all

% Load Training data and Validation Method (k-value)
load('EvaluationMethodParameter.mat','k','data','cvp');
data_2=data;

% Assume Tree no=1500
rng(1); % reproducibility
mdlRF_TreeNo = fitensemble(data_2,'BikeBuyer','Bag',1500,'Tree', 'Type','Classification','CVPartition',cvp,'Resample','on');
mdlRF_TreeNo_b = fitensemble(data_2,'BikeBuyer','Bag',1500,'Tree', 'Type','Classification','Resample','on');


clossRF_1=kfoldLoss(mdlRF_TreeNo,'mode','cumulative');
clossRF_1b=oobLoss(mdlRF_TreeNo_b,'mode','cumulative');
[clossRF_1min,tidx]=mink(clossRF_1,20); % Check min loss and optimal tree no. if the error rate is stable
[clossRF_1bmin,tidxb]=mink(clossRF_1b,20); 

% Plot loss with no. of tree grown for visualization
figure;
cla reset
plot(1:1500,clossRF_1,'b-','LineWidth',1.5);
hold on
plot(1:1500,clossRF_1b,'g--','LineWidth',1.5);
plot(tidx(1),clossRF_1min(1),'Color','r','Marker',"o",'MarkerSize',10)
plot(tidxb(1),clossRF_1bmin(1),'Color','r','Marker',"o",'MarkerSize',10)

xlabel('Number of Learning Cycles');
ylabel('Classification error')
ylim([0.22 0.27])
title('Classification Error Vs. Number of Learning Cycling of Random Forest')
legend({'K-Fold Validation','Out-Of-Bag Validation','Min. Error','Min. Error'},'Location','NE');
hold off;

% Generalization loss becomes stable since tree =800
% More variations among Out-of-bag loss

%%  Baseline Random Forest Mondel
clear all

% Load Training data and Validation Method (k-value)
load('EvaluationMethodParameter.mat','k','data','cvp');
data_2=data;
tree=800;

% Construct baseline model
rng(1); % reproducibility
t=templateTree('Reproducible',true);
mdlRF_baseline = fitensemble(data_2,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','CVPartition',cvp,'Resample','on','Learners',t);
% Loss
clossRF_2=kfoldLoss(mdlRF_baseline,'mode','cumulative');
[clossRF_2min,tidx2]=min(clossRF_2);

%% Predictor Importance - insight in feature selection 
% Estimate predictor importance
rng(1);
t2=templateTree('Reproducible',true,'Surrogate','on','PredictorSelection','curvature'); % for unbiased test
mdlRF=fitensemble(data_2,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t2);
[imp,bidx]=predictorImportance(mdlRF);

% Plot graph on predictor importance
[imp_rank,idx]=maxk(imp,14); % sort importance descendingly
figure
cla reset
name=data_2.Properties.VariableNames(idx);
bar(categorical(name),imp_rank)
title('RF Predictor Importance Estimates for Bike-Buyer Classification')
xlabel('Predictors')
ylabel('Estimates')
saveas(gcf,'Chart\PredictorImportance.jpg')


%% Dimension reduction and evaluation

% Removal of Predictors for improving performance
loss_predictor = zeros(k,14);

for i=1:14
    colidx=idx(1:i);
    df=data_2(:,[colidx,15]);
    rng(1); % reproducibility
    t=templateTree('Reproducible',true);
    mdl = fitensemble(df,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','CVPartition',cvp,'Resample','on','Learners',t);
    loss_predictor(:,i)=kfoldLoss(mdl,'mode','individual');
end

% Plot graph
figure 
cla reset
% error bar
pos=max(loss_predictor)-mean(loss_predictor);
neg=mean(loss_predictor)-min(loss_predictor);

errorbar(1:14,mean(loss_predictor),neg,pos,'LineWidth',1,'Marker','o','Color','b');
xlabel('Number of Feature Kept in Model')
ylabel('Generalization Error')
title('Features included in Model Tuning')
saveas(gcf,'Chart\FeatureSelection.jpg');
[~,I]=min(mean(loss_predictor));
data_3=data_2(:,[1:I,15]);

save('RF_Parameter_Assessment\RF_FeatureSelection.mat','I','data_3','loss_predictor','imp_rank','idx','imp','tree');

%% Spilt Criterion

clear all
load('EvaluationMethodParameter.mat','k','cvp');
load('RF_FeatureSelection.mat','data_3','tree')

criteria={'gdi','deviance'};

% Train model on different split criterion
loss_criteria=zeros(k,2);
trainloss_criteria=zeros(1,2);
AUC_criteria=zeros(1,length(criteria));
F1_score_criteria =zeros(1,length(criteria));

for i=1:2
    rng(1);
    t=templateTree('Reproducible',true,'SplitCriterion',criteria{i});
    mdl=fitensemble(data_3,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t);
    model = crossval(mdl,'CVPartition',cvp);
    loss_criteria(:,i)=kfoldLoss(model,'mode','individual');
    trainloss_criteria(i)=resubLoss(mdl);
    [pred,score]=kfoldPredict(model);
    [~,~,~,AUC]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');
    F1_score_criteria(i)=f1score(data_3.BikeBuyer,pred);
    AUC_criteria(i)=AUC;         
end

% Plot error chart
figure
cla reset
bar(categorical(criteria),F1_score_criteria)
bar(categorical(criteria),AUC_criteria)
hold on
plot(categorical(criteria),mean(loss_criteria))
plot(categorical(criteria),mean(trainloss_criteria))
hold off
saveas(gcf,'Chart\BestSplitCriterion.jpg')
spiltcrit='deviance';
save('RF_Parameter_Assessment\BestSpiltCriterion.mat','loss_criteria','F1_score_criteria','AUC_criteria','spiltcrit')

%% Number of Feature to Sample per node

clear all
load('EvaluationMethodParameter.mat','k','cvp');
load('RF_FeatureSelection.mat','data_3','tree');
load('BestSpiltCriterion.mat','spiltcrit');

m=14; % number of predictors
no=ceil(log2(m)+1); % Calculation suggested by Scholars for high dimensional dataset

% Empty list to store loop info
loss_spilt=zeros(tree,m);% K-fold * Predictor Numbers
trainloss_spilt=zeros(1,m);% K-fold * Predictor Numbers
AUC_spilt=zeros(1,m); 
F1_score_spilt=zeros(1,m);
time_spilt=zeros(1,m); 

%loop 
for i=1:m
    tic
    t=templateTree('Reproducible',true,'SplitCriterion',spiltcrit,'NumVariablesToSample',i);
    model = fitensemble(data_3,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t);
    model_CV=crossval(model,'CVPartition',cvp);
    time_spilt(i)=toc;
    loss_spilt(:,i)=kfoldLoss(model_CV,'mode','cumulative');
    trainloss_spilt(i)=resubLoss(model);
    [pred,score]=kfoldPredict(model_CV);
    [~,~,~,AUC]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');
    AUC_spilt(i)=AUC;
    F1_score_spilt(i)=f1score(data_3.BikeBuyer,pred);
end

% plot No. of Feature Vs. No. Tree Vs. Validation Loss (3-Dimensional)
figure 
cla reset
[x_surf_spilt,y_surf_spilt]=meshgrid(1:14,1:800);
surf(x_surf_spilt,y_surf_spilt,loss_spilt)
colormap(summer)
colorbar()
title('Number of Feature to Sample Tuning')
xlabel('No of Feature')
ylabel('No. of Trees Grown')
view([23.3722 3.7073])
saveas(gcf,'Chart\No_Feature_Per_Spilt_3D.jpg')

% Plot No. of Feature Vs. all perforamnce matrix
figure
cla reset
yyaxis left
plot(1:m,mean(loss_spilt),'Marker','o','MarkerEdgeColor','auto','LineWidth',1)
hold on
plot(1:m,trainloss_spilt,'Marker','*','Color','g','LineWidth',0.5)
title('Number of Feature to Sample Tuning')
xlabel('Number of predictors to consider at random for each split')
ylabel('Generalization Error')
xlim([0 m+1])
ylim([0 0.3])
hold off

yyaxis right
plot(1:m,AUC_spilt,'LineWidth',1)
ylabel('AUC of PRC curve')
legend({'Validation Error','Training Error','Validation AUC'},'location','best')

minloss_spilt=min(loss_spilt,[],'all');
[~,No_feature_spilt]=find(loss_spilt==minloss_spilt);
saveas(gcf,'Chart\DT_No_Feature_Per_Spilt.jpg')
save('RF_Parameter_Assessment\Feature_Per_Spilt.mat','No_feature_spilt','loss_spilt','AUC_spilt','trainloss_spilt','time_spilt','F1_score_spilt')

%% Maximum No of Nodes to Spilt and Min. Leaf Size to control Tree Depth
clear all
load('EvaluationMethodParameter.mat','k','cvp');
load('RF_FeatureSelection.mat','data_3','tree');
load('BestSpiltCriterion.mat','spiltcrit');
load('Feature_Per_Spilt.mat','No_feature_spilt')
[sample,~]=size(data_3); % No of observations
m = floor(log2(sample - 1)); 
maxSplits = 2.^(0:m);
leafsize=[1,5:5:100];

% empty list to store loop info
loss_nodespilt=zeros(length(leafsize),length(maxSplits));
AUC_nodespilt=zeros(length(leafsize),length(maxSplits));
F1_score_nodespilt=zeros(length(leafsize),length(maxSplits));
trainloss_nodespilt=zeros(length(leafsize),length(maxSplits));
time_nodespilt=zeros(length(leafsize),length(maxSplits));

% looping through leaf size and node number

for i=1:length(maxSplits)
    rng(1);
    nodespilt=maxSplits(i)
    for j=1:length(leafsize)
        ls=leafsize(j);
        t=templateTree('Reproducible',true,'SplitCriterion',spiltcrit,'NumVariablesToSample',No_feature_spilt,'MaxNumSplits',nodespilt,'MinLeafSize',ls);
        tic
        model = fitensemble(data_3,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t);
        time_nodespilt(j,i)=toc;
        model_CV=crossval(model,'CVPartition',cvp);
        loss_nodespilt(j,i)=kfoldLoss(model_CV);
        trainloss_nodespilt(j,i)=resubLoss(model);
        [pred,score]=kfoldPredict(model_CV);
        [~,~,~,AUC]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');
        AUC_nodespilt(j,i)=AUC;  
        F1_score_nodespilt(j,i)=f1score(data_3.BikeBuyer,pred);
    end   
end

%% Identify best performance
vl_min=min(loss_nodespilt,[],'all');
[vl_x,vl_y]=find(loss_nodespilt== vl_min);
tl_min=min(time_nodespilt,[],'all');
[tl_x,tl_y]=find(time_nodespilt== tl_min);
f1_max=max(F1_score_nodespilt,[],'all');
[f1_x,f1_y]=find(F1_score_nodespilt== f1_max);
auc_max=max(AUC_nodespilt,[],'all');
[auc_x,auc_y]=find(AUC_nodespilt== auc_max);

%% Plot Surface graph
[x_surf,y_surf] = meshgrid(maxSplits,leafsize);
figure
cla reset

% Validation Loss
subplot(3,2,1);
surf(x_surf,y_surf,loss_nodespilt);
title('Validation Data Loss')
zlabel('Loss')
view([-33.177 10.476])

% Training Loss
subplot(3,2,2);
surf(x_surf,y_surf,trainloss_nodespilt);
colormap(winter)
zlabel('Loss')
title('Training Data Loss')
view([53.732 19.536])

% Time spend per model
subplot(3,2,3);
surf(x_surf,y_surf,time_nodespilt);
title('Training Time')
zlabel('Training Time')

% F1 Score
subplot(3,2,4);
surf(x_surf,y_surf,F1_score_nodespilt);
zlabel('F1 Score')
title('F1 Score')

% AUC under PRC
subplot(3,2,5);
surf(x_surf,y_surf,AUC_nodespilt);
zlabel('AUC')
title('AUC of Precision Recall Curve')

subplot(3,2,6);
plot(x_surf,y_surf)
hold on
plot(maxSplits(vl_y),leafsize(vl_x),'Marker','diamond','MarkerFaceColor','r','MarkerSize',10)
plot(maxSplits(f1_y),leafsize(f1_x),'Marker','hexagram','MarkerFaceColor','g','MarkerSize',10)
plot(maxSplits(tl_y),leafsize(tl_x),'Marker','o','MarkerFaceColor','k','MarkerSize',10)
plot(maxSplits(auc_y),leafsize(auc_x),'Marker','pentagram','MarkerFaceColor','b','MarkerSize',10)
hold off
xlabel('Max. Branch node')
ylabel('Min Leaf Size')
xlim([0 1200])
title('Best Performance')


% Optimal parameter
BestMaxNumSplit=maxSplits(vl_y);
BestMinLeaf=leafsize(vl_x);
Time=time_nodespilt(vl_x,vl_y);
saveas(gcf,'Chart\TreeDepth.jpg')
save('RF_Parameter_Assessment\TreeDepth.mat','loss_nodespilt','AUC_nodespilt','F1_score_nodespilt','trainloss_nodespilt','time_nodespilt','leafsize','maxSplits','m','BestMinLeaf','BestMaxNumSplit')

%% Class Prior Tuning

%If The class prior probabilities (Prior) are highly skewed, the software oversamples unique observations from the class that has a large prior probability.
% Especially when AUC shows better result in NB
clear all
load('EvaluationMethodParameter.mat','k','cvp');
load('RF_FeatureSelection.mat','data_3','tree');
load('BestSpiltCriterion.mat','spiltcrit');
load('Feature_Per_Spilt.mat','No_feature_spilt');
load('TreeDepth.mat','BestMaxNumSplit','BestMinLeaf');

priors=0.1:0.05:0.9;

% Empty list to store loop info
trainloss_prior = zeros(1,length(priors));
loss_prior = zeros(k,length(priors));
F1_score_prior = zeros(1,length(priors));
AUC_prior = zeros(1,length(priors));
time_prior = zeros(1,length(priors));

% Loop model to tune
for i=1:length(priors)
    p=priors(i);
    t=templateTree('Reproducible',true,'SplitCriterion',spiltcrit,'NumVariablesToSample',No_feature_spilt,'MaxNumSplits',BestMaxNumSplit,'MinLeafSize',BestMinLeaf);
    tic
    model = fitensemble(data_3,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t,'ClassNames',[0,1],'Prior',[1-p p]);
    model_CV=crossval(model,'CVPartition',cvp);
    time_prior(i)=toc;
    trainloss_prior(i)=resubLoss(model);
    loss_prior(:,i)=kfoldLoss(model_CV,'mode','individual');
    [pred,score]=kfoldPredict(model_CV);
    [~,~,~,AUC]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');
    F1_score_prior(i)=f1score(data_3.BikeBuyer,categorical(pred));
    AUC_prior(i)=AUC;        
end

% Plot graph
% error bar 
pos=max(loss_prior)-mean(loss_prior);
neg=mean(loss_prior)-min(loss_prior);

figure
cla reset
yyaxis left
errorbar(priors,mean(loss_prior),neg,pos,'Marker','o','Color','b')
hold on
plot(priors,trainloss_prior,'Marker','*','Color','g')
hold off
ylabel('Loss')

yyaxis right
plot(priors,AUC_prior,'m--','LineWidth',1)
hold on
plot(priors,F1_score_prior,'r--','LineWidth',1)
ylabel('Score')
hold off

xlabel('Prior')
title('Class Prior Tuning')
legend({'Validation Error','Training Error','AUC of PRC','F1 Score'},'location','best')

saveas(gcf,'Chart\RF_ClassPrior.jpg');

BestPrior=0.73; % Trade off of F1 score and validation loss
save('RF_Parameter_Assessment\RF_Prior.mat','time_prior','priors','trainloss_prior','loss_prior','F1_score_prior','AUC_prior','BestPrior')


%% Final Model of Random Forest
clear all
load('RF_FeatureSelection.mat','data_3','tree');
load('BestSpiltCriterion.mat','spiltcrit');
load('Feature_Per_Spilt.mat','No_feature_spilt')
load('TreeDepth.mat','BestMaxNumSplit','BestMinLeaf')
load('RF_Prior.mat','BestPrior')
load('EvaluationMethodParameter.mat','k','cvp');

t=templateTree('Reproducible',true,'SplitCriterion',spiltcrit,'NumVariablesToSample',No_feature_spilt,'MaxNumSplits',BestMaxNumSplit,'MinLeafSize',BestMinLeaf);
tic
RF_model_Final = fitensemble(data_3,'BikeBuyer','Bag',tree,'Tree', 'Type','Classification','Resample','on','Learners',t,'ClassNames',[0,1],'Prior',[1-BestPrior BestPrior]);
RF_model_Final_CV=crossval(RF_model_Final,'CVPartition',cvp);
Final_RF_time=toc;
save('Model_RF_parameter.mat','t','RF_model_Final','BestPrior','tree','spiltcrit','No_feature_spilt','BestMaxNumSplit','BestMinLeaf')

%% Validation Test on Final Model

TrainLoss_Final=resubLoss(RF_model_Final);
kLoss_Final=kfoldLoss(RF_model_Final_CV);
[pred,score]=kfoldPredict(RF_model_Final_CV);

F1_Validation=f1score(data_3.BikeBuyer,categorical(pred));
[~,~,~,AUC_Validation]=perfcurve(data_3.BikeBuyer,score(:,2),'1','XCrit',"reca",'YCrit','prec');

% Save Model and Validation Result
save('RF_Parameter_Assessment\RF_Validation_Result.mat','Final_RF_time','TrainLoss_Final','kLoss_Final','F1_Validation','AUC_Validation')
