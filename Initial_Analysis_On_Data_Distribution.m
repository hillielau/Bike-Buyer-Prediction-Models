%% Understand the Dataset Prior to Model Selection 

clear all
clc

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

% Evaluate the Dataset Balance
tabulate(bike.BikeBuyer) %Imbalance Dataset
% Plot Bike Buyer Distribution
figure
cla reset
histogram(bike{:,15},"Normalization","probability");
title('Bike Buyer')
xticklabels({'No','Yes'})
ylabel('Probability')
% Summary: Dataset is slightly imbalance with 66% non-buyers

% Initial Analysis on Data Distribution
idx=bike.BikeBuyer=='0';
stat = zeros(6,6);

%% Evaluate the distribution of numeric predictors
for i = 9:14
    
    stat(i-8,1)=mean(bike{~idx,i});
    stat(i-8,2)=mean(bike{idx,i});
    stat(i-8,3)=std(bike{~idx,i});
    stat(i-8,4)=std(bike{idx,i});
    stat(i-8,5)=skewness(bike{~idx,i},0,1);
    stat(i-8,6)=skewness(bike{idx,i},0,1);
    
    
    % show histogram by Buyer Type and fit normal distribution
    figure
    bin_i=20;
    histfit(bike{idx,i},bin_i,'normal');  % check normality
    hold on
    histfit(bike{~idx,i},bin_i,'normal');
    title(bike.Properties.VariableNames(i)+" by Bike Buyers");
    legend('Non-Buyers','normal curve','Bike Buyers');
    hold off

    figure
    histfit(bike{:,i},20,'normal')
    title(bike.Properties.VariableNames(i));
   
end

% Summary
% 1) Age & Monthly Spending in company shows slightly negatively skewed distribution,
% 2) Age of bike buyers are normally distributed
% 3) Age does not show noticeable difference between distributions of two classes
% 4) Yearly Income are normally distributed.
% 5) Distributions of two classes overlays 
% 6) Some numeric predictors, as non-negative integers, are pre-dominant by 0

%%  Distribution of Categorical predictors
for col = (1:8)
    figure
    histogram(bike{~idx,col},"Normalization","probability");
    hold on
    histogram(bike{idx,col},"Normalization","probability",'BarWidth',0.5);
    title(bike.Properties.VariableNames(col));
    legend({'Buyer','Non-buyer'},'location','best')
    ylabel('Probability');
    hold off
end

% Summary
% 1) Uniform distributions observed among Martial Status & Gender
% 2) Home Owner Flag shows same distribution among buyer and non-buyer
% 3) Pre-dominance of a few locations in City/ State Province 





