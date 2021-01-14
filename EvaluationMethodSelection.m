%% Investigate suitable Method for Evaluation (Hold Out Vs. Cross Validation)

% load Bike Dataset
bike=readtable('Data\BikeMergeClean.csv');
% drop CustomerID from data
bike=bike(:,2:end);

% Convert the Categorical data into Categorical data
bike.City=categorical(bike.City);
bike.StateProvinceName=categorical(bike.StateProvinceName);
bike.CountryRegionName=categorical(bike.CountryRegionName);
bike.Education=categorical(bike.Education);
bike.Occupation=categorical(bike.Occupation);
bike.Gender=categorical(bike.Gender);
bike.MaritalStatus=categorical(bike.MaritalStatus);
bike.HomeOwnerFlag=categorical(bike.HomeOwnerFlag);
bike.BikeBuyer=categorical(bike.BikeBuyer);

%% Data Partition - Hold Out Validation Dataset For Method Comparison
rng(1); % For reproducibility
cvpt=cvpartition(bike.BikeBuyer,'Holdout',0.2,'Stratify',false); %0.2 as suggested by scholars
data=bike(cvpt.training,:); % Train dataset
dataValidation=bike(cvpt.test,:); % Validation dataset for Final Testing

% Understand Imbalance dataset
tabulate(dataValidation.BikeBuyer)
tabulate(data.BikeBuyer)

%% Independent Hold Out Validation
% Compare Validation Method on data partitioning and evaluate the loss with baseline models 

ratelist=0.1:0.1:0.9;
NBilist=zeros(1,length(ratelist));
RFilist=zeros(1,length(ratelist));

for i = 1:length(ratelist)
    rng(1);
    rate=ratelist(i);
    cvpI=cvpartition(data.BikeBuyer,'HoldOut',rate,'Stratify',false);
    trainI=data(cvpI.training,:);
    testI=data(cvpI.test,:);
    
    % Model built with holdout dataset
    mdlNBi = fitcnb(trainI,'BikeBuyer');
    mdlRFi = fitensemble(trainI,'BikeBuyer','Bag',100,'Tree','Type','Classification');
    
    % Loss 
    NBilist(i)=loss_spilt(mdlNBi,testI);
    RFilist(i)=loss_spilt(mdlRFi,testI);
end

% Plot Error of Hold-Out Partition
[hmin,hidx]=min((NBilist + RFilist)/2);
figure;
cla reset
plot(ratelist,NBilist,'LineWidth',1,'Color','g','LineStyle',"--")
hold on
plot(ratelist,RFilist,'LineWidth',1,'Color','r','LineStyle',":")
plot(ratelist,(NBilist + RFilist)/2,'Color','k','LineStyle',"-.",'LineWidth',1)
hold off
xlabel('Hold Out%')
ylabel('Error')
legend({'Naive Bayes Baseline','Random Forest Baseline','Average'})
Bestrate=ratelist(hidx);
title('Hold Out Partition | Best: ' +string(Bestrate)+' @'+string(hmin))

saveas(gcf,'Chart\HoldOutLoss.jpg');



%% K-Fold Cross Validation

% Set up K value list
klist= 5:1:20;

% Construct empty error matrix
NBklist=zeros(length(klist),4);
RFklist=zeros(length(klist),4);

% Test K value for each method - WARNING: Time-Consuming
for i = klist
    rng(1);
    cvpkf=cvpartition(data.BikeBuyer,'KFold',i);
    
    % K-Fold Cross Validation Model
    mdlNBkf = fitcnb(data,'BikeBuyer','CVPartition',cvpkf);
    % Assume Tree Number = 100
    mdlRFkf = fitensemble(data,'BikeBuyer','Bag',100,'Tree','CVPartition',cvpkf,'Type','Classification');
    
    % Obtain the error within each fold
    lossNBkf= kfoldLoss(mdlNBkf,'Mode','individual','LossFun','classiferror');
    NBklist(i-4,:)=[mean(lossNBkf), min(lossNBkf),max(lossNBkf),std(lossNBkf)] ;
    lossRFkf = kfoldLoss(mdlRFkf,'Mode','individual','LossFun','classiferror');
    RFklist(i-4,:)=[mean(lossRFkf), min(lossRFkf),max(lossRFkf),std(lossRFkf)] ;
end



%% Compare errors of two K-fold Models

% Display the mean and variance of each K-fold with Error Bar

% Construct error bar
errNB= NBklist(:,2:3)-NBklist(:,1); 
errRF= RFklist(:,2:3)-RFklist(:,1);

% Combine and Average mean and variance of two k-fold models
neg=(-errNB(:,1) -errRF(:,1))/2';
pos=(errRF(:,2)+ errNB(:,2))/2';
errmean=(RFklist(:,1)+NBklist(:,1))/2';

% Construct figure
k1=figure;
errorbar(klist,NBklist(:,1)',-errNB(:,1)',errNB(:,2)','LineStyle',"none",'Marker','o','Color','b','LineWidth',1)
hold on
errorbar(klist,RFklist(:,1)',-errRF(:,1)',errRF(:,2)','LineStyle',"none",'Marker','*','Color','g','LineWidth',1,"CapSize",10)
xlim([0,25])
xlabel('K-fold')
title('Mean and Variance of K-Fold Loss by Method')
ylabel('Error')
legend({'Naive Bayes','Random Forest'},"Location","best")
hold off

% Plot the Average Mean variance and  Error of the two models
k2=figure;
errorbar(klist,errmean,neg,pos,'LineStyle',"none",'Marker','o','Color','k','LineWidth',1)
hold on
[kmin,kidx]=min(errmean);
[~,vidx]=min(pos+neg);
plot(klist(kidx),kmin,'Color','r',"Marker",'o','MarkerSize',12)
errorbar(klist(vidx),errmean(vidx),neg(vidx),pos(vidx),'LineStyle',"none",'Marker','none','Color','m','LineWidth',1)
hold off
title('Mean and Variance of K-Fold Loss - Average')
legend({'Error bar','Min. Mean Error','Min Variance'},'Location',"best")
xlim([0,25])
xlabel('K-fold')

%% Trade off on error and variance - Compare

% K-Fold Validation present a more stable error rate in validation

% Compare Error rate and variance for best choice of K-value
errsort=sortrows([klist',errmean,pos+neg],2,"ascend");
varsort=sortrows([klist',errmean,pos+neg],3,"ascend");

% Decision on k value
k=8;
disp('Best Cross Validation: k = 8')
disp('Mean Error= ' + string(errsort(3,2)))

saveas(k1,'Chart\kFoldLossByMethod.jpg');
saveas(k2,'Chart\kFoldLoss.jpg');

% Baseline Model Using K-fold Validation k=8
rng(1)
cvp=cvpartition(data.BikeBuyer,'KFold',k);

% Save  the Validation Method
save('EvaluationMethodParameter.mat','k','ratelist','NBilist','RFilist','hmin','Bestrate','klist','NBklist','RFklist','errmean','errNB','errRF','data','dataValidation','cvp');

