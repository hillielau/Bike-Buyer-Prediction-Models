# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:34:37 2020

@author: Hillie
"""

import pandas as pd
import datetime as dt



file=pd.read_csv('AdvWorksCusts.csv',parse_dates=['BirthDate'])
                 

print(file.info())

buyer=pd.read_csv('AW_BikeBuyer.csv')
buyer=buyer.sort_values(by=['CustomerID','BikeBuyer'])



print(buyer[buyer.duplicated(subset=['CustomerID','BikeBuyer'])])


buyer=buyer.drop_duplicates(keep='first',subset=['CustomerID','BikeBuyer'])

## Remove Customers identified both as Bike buyer and non-bike buyer
print(buyer[buyer.duplicated(subset=['CustomerID'],keep=False)])
buyer=buyer.drop_duplicates(subset='CustomerID',keep=False)

print(buyer)

print(file.shape)


df=pd.merge(file,buyer,on='CustomerID',how='inner',sort=True);
print(df.shape)

print(df.info())
print(df.head(5))
df['City']=df['City'].str.lower()

# Remove columns showing only unqiue info
df['Age']=1998-df['BirthDate'].dt.year
column_delete=['Title','FirstName','MiddleName','LastName','Suffix','AddressLine1','AddressLine2','PhoneNumber','BirthDate','PostalCode']

df.drop(axis=1,labels=column_delete,inplace=True)
df=df.sort_values(by=['CustomerID','YearlyIncome','TotalChildren']).drop_duplicates(keep='last',subset=['CustomerID'])

print(df.info())




#####################################################

city=pd.read_csv('worldcities.csv',sep=',')
city=city[['city_ascii','lat','lng','population']]
city.columns=['City','Latitude','Longitude','Population']

city=city.dropna(axis=0,how='any')
city['City']=city['City'].str.lower()

print(city.isnull().sum())
print(city.info())
print(city.head(10))
print(city[city.duplicated(subset='City',keep=False)].sort_values(by='City'))

city=city.drop_duplicates(subset='City',keep=False)

##########################################################################
df=pd.merge(df,city,on='City',sort=True,how='left')
df.reset_index()



############################################
expense=pd.read_csv('AW_AveMonthSpend.csv',sep=',')
expense=expense.drop_duplicates(keep='first')

dup=expense[expense.duplicated(subset='CustomerID',keep=False)].sort_values(by='CustomerID')

dup_g=pd.DataFrame(dup.groupby('CustomerID')['AveMonthSpend'].mean())
dup_g.reset_index(level=0,inplace=True)

expense=expense.drop_duplicates(keep=False,subset='CustomerID')
expense=pd.concat([expense,dup_g],axis=0)

print(expense[expense.duplicated(subset='CustomerID',keep=False)].sort_values(by='CustomerID'))

df=pd.merge(df,expense,on='CustomerID',how='left')


#############################################
idx=df.columns.get_loc('BikeBuyer')


list=[i for i in range(idx)] +[i for i in range(idx+1,df.shape[1])]+[idx]

df=df.iloc[:,list]


df=df.dropna(axis=1,how='any')
df=df.drop(columns=['CustomerID'],axis=1)

print(df.shape)
df.to_csv('BikeMergeClean.csv',sep=',')

