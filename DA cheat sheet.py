######################################### PACKAGE IMPORTS ##################################################################

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib inline

# this import allows us to read stock market data from google and yahoo
from pandas_datareader import data, wb

# this will allow us to set the date and time of our data grab
from datetime import datetime

# this is imported so that we dont have to worry about including floats in our division operations
from __future__ import division

# requests is imported to gather data from the web
import requests

# StringIO will make it easier to read csv data
from StringIO import StringIO

######################################### DATA IMPORT #######################################################################

# import a csv in the same folder as the jupyter notebook. if not in same folder then insert destination
titanic_df = pd.read_csv('train.csv')
titanic_df = pd.read_csv('C:\Users\munaw\Desktop\Python\Titanic\data.csv')

#################################### EXPLORATORY DA #########################################################################

# to get the shape of a dataset
occupation_df.shape

# display the first 5 rows of the data set
titanic_df.head()

# display the last 5 rows of the data set
titanic_df.tail()

# note that a number can be entered in the parenthesis to view more rows, this works with .head() and .tail()
titanic_df.head(20)
titanic_df.tail(15)

# .head() can also be called on a series or variable ('deck), and it will display the first 5 rows of data
deck.head() 

# this will display the number of columns in the data set, name of each column, the number of entries in each column,
# and the data type of each column
titanic_df.info()

# this gives a quick statistical summary of every column. It will show the count, mean, std, min, max, and 25/50/75 
# percentiles for every column in a dataset. Here, 'AAPL' is the name of the data set, not the name of a column.
AAPL.describe()

# .describe() can also be used on a particular column
titanic_df['Fare'].describe()

# you can also use describe on specific columns only by passing a list to the argument
list = ['Fare','Age']
titanic_df[list].describe()

# display all unique values of a dataset
stocks_df['Name'].unique()

# value counts for a particular column
titanic_df['person'].value_counts()

# combining value counts and head()
top_donor.value_counts().head(10)

# show the unique values in a list
candidates = donor_df.cand_nm.unique()

# finding the sum of null values of all columns in a pandas dataframe. 'data' is the name of the dataset in this case
data.isnull().sum()

#################################### GRAHPS ################################################################################

# the simplest way to make a chart. you can change 'bar' to other types, play around with it
cand_amount.plot(kind='bar')

# this will just set the background to graphs as a white grid
sns.set_style('whitegrid')

# a bar chart (countplot) comparing the total values of all values within a variable (sex)
sns.countplot('Sex',data=titanic_df)

# you can further breakdown a count plot by specifying a hue
sns.countplot('Sex',data=titanic_df, hue='Pclass')

# you can create countplot with sorted values, and adjust the color palette. the palette is not a required entry, 
# but here it is chosen to be a specific type. palettes can be found here: https://matplotlib.org/users/colormaps.html
# you can also choose a palette name and add '_d' to the end to make it darker. for example, 'hot_d'
# the cabin values wont be in abc order unless you state the .values.sort() code
cabin_df['Cabin'].values.sort()
sns.countplot('Cabin',data=cabin_df,palette='hot')

# you can also sort the data in the following way. you can pass only the value you wish to see in a list in the
# 'order' parameter.
sns.countplot('Embarked', data=titanic_df, hue='Pclass', order=['C','Q','S'],palette='hot')

# this code will eliminate 'T' from the cabin_df dataframe all together, and then create a count plot.
cabin_df = cabin_df[cabin_df.Cabin != 'T']
cabin_df['Cabin'].values.sort()
sns.countplot('Cabin',data=cabin_df,palette='brg')

# this create a histograme of the given variable, with 70 bins
titanic_df['Age'].hist(bins=70)

# a FacetGrid is used to create multiple plots on one figure. Here, multiple plots of 'Sex' will be plotted on the 
# same figure. In particular, kde plots of the Age will be shown, broken down by 'Sex' which is the hue. 
# kde plots visualize the distribution of data over an interval, are variations of histograms. However, they are 
# better at determining distirbution shape, and do not rely on bin size to create smooth distributions. 
# Aspect changes the size of the plot, and shade will shade in beneath each of the lines. note how the x-limits are set
# and how a legend is added to the figure.
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# a factorplot (also called 'catplot') shows the relationship and regression line between two variables,
# 'Pclass' and 'Survived', and can also take into account a hue
sns.factorplot('Pclass','Survived',data=titanic_df,hue='person')

# lmplot plots data and regression model fits across a facetgrid (slightly different than lmplot)
sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass')

# this will create an lmplot but with specified bins
generations = [10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='Set1',x_bins=generations)

# a regular plot of a specific column ('Adj Close') of a dataset ('AAPL'). legend and figsize are optional inputs
AAPL['Adj Close'].plot(legend='True',figsize=(10,4))

# another standard plot, but with multiple columns in one plot. Note that a subplot can be created by 
# setting subplot to 'True'
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))

# a distplot is a distribution line plotted over a histogram to show a distribution of values. Here, the
# null values are also dropped.
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')

# a joint plot plots two variables or the same variable but for two different entities/data tables/columns to
# show the correlation. Note that 'reg', 'resid', 'kde', and 'hex' can also be used in place of 'scatter'.
# In this case, 'GOOG' and 'MSFT' are two different columns in the same table.
sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter',color='red')

# pairplot will plot all the columns against each other to get a general view of all the different correlations.
# pairplot will plot a grid of all the different column pairs and their correlations. note here that null
# values are dropped.
sns.pairplot(tech_rets.dropna())

# an alternate to pairplot is pairgrid. PairGrid can be used for full control over the grid, including the 
# upper and lower triangles and the diagonal.
returns_fig = sns.PairGrid(tech_rets.dropna())
# this will determine what the upper triangle will look like
returns_fig.map_upper(plt.scatter,color='orange')
# choose a different type of plot for lower triangle
returns_fig.map_lower(sns.kdeplot,cmap='hot_d')
# now to adjust the diagonal
returns_fig.map_diag(plt.hist,bins=30)

# this will display a grid of correlations, but a heatmap grid
sns.heatmap(tech_rets.dropna().corr(),annot=True)

# q here is the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)

# below are all uses of 'plt' and 'figtext' to edit a graph. note the creation of a vertical line below.
# the vertical line is plotted at a point on the x-axis that you determine.
# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)
# Using plt.figtext to fill in some additional information onto the plot
# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())
# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))
# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
# Plot a vertical line at the 1% quantile result.
plt.axvline(x=q, linewidth=4, color='r')
plt.axvline(x=327, linewidth=4, color='blue')
# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');

# this creates a bar chart with error lines in each bar. Here, 'avg' is the mean which was calculated
# previously and is a variable, 'std' is a previously calculated std and is also a variable.
avg.plot(yerr=std,kind='bar',legend=False)

# a time series plot where the x and y inputs are given
time_series = poll_df.plot(x='End Date', y=['Obama','Romney','Undecided'],marker='o',linestyle='',figsize=(10,5))

# this is the same thing as above but with x-limits set
diff_plot = poll_df.plot(x='Start Date', y='Difference', marker='o', linestyle='-', color='red', figsize=(12,5),
                         xlim=(min(xlimit),max(xlimit)))

# invert the axis of a graph. here, 'time_series' is the name of the graph/variable
time_series.invert_xaxis()

# plotting a chart after using .groupby() and .sum(). 'Party' is what is being grouped by, and 'contb_receipt_amt' is
# what is being summed.
donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar',color=['blue','red'])

# horizontal bar chart
occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')

#################################### MATHEMATICAL CACULATIONS ############################################################

# find the mean of any column
titanic_df['Age'].mean()

# find the max and min of any column
titanic_df['Age'].max()
titanic_df['Age'].min()

# group by along with count and sum of a particular column
donor_df.groupby('cand_nm')['contb_receipt_amt'].count()
donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

# standard deviation
std = pd.DataFrame(poll_df.std())

# now lets calculate the rolling mean, which is the mean over the course of the past x amount of days
# adjust each day. so a 5 day rolling mean is the mean of a value over the past 5 days, and for the next
# day the 5 day period is adjust such that the previous last day is dropped.
ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))  #this will set the contents of ma_day to a string
    AAPL[column_name]=AAPL['Adj Close'].rolling(window=ma).mean() #this will create a column in AAPL equal to
                                                                  #the rolling mean, which is a pandas function

# this will calculate a percent change. A new column ('Daily Return') is added to AAPL dataset and is set equal
# to the percent change of the 'Adj Close' column.
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

# The 0.05 empirical quantile of daily returns. # The 0.05 empirical quantile of daily returns is at -0.0315. 
# That means that with 95% confidence, our worst daily loss will not exceed 3.15%. If we have a 1 million dollar 
# investment, our one-day 5% VaR is 0.0315 * 1,000,000 = $31,500. 95% of the times, the maximum amount of money 
# we'll lose is 3.15% of our investment. only 5% of the times we'll lose more.
rets['AAPL'].quantile(0.05)

# finding the difference between values of two different columns
poll_df['Difference'] = (poll_df.Obama - poll_df.Romney)/100

# this will group the dataset by a certain column (same thing as group by in sql). you can then decide how to display
# the grouped information, and here the mean of the grouped values is shown.
poll_df = poll_df.groupby(['Start Date'], as_index=False).mean()


################################## other #################################################################################

# this will set the end as the current date, and the start as the current month and day of the previous year
end = datetime.now()
start = datetime(end.year-1,end.month,end.day)

################################## DATA MANIPULATION #####################################################################

# this will create a new column within the existing dataframe. The new column will take in 'age' and 'sex' and apply it
# to the 'male_female_child' function (below), and the result of the function will be the value of the column. 
# remember to change the axis to 1.  
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

# a column from an existing dataframe ('Cabin') is being copied to a new variable ('deck'), and all null values 
# are being dropped.
deck = titanic_df['Cabin'].dropna()

# this will concatenate two previously existing columns into a dataframe
poll_avg = pd.concat([avg,std],axis=1)

# this will create a new dataframe ('cabin_df') from an existing list ('levels')
cabin_df = DataFrame(levels)

# this will name the columns of the df, i think you can pass a list to name all columns
cabin_df.columns = ['Cabin']

# this code will eliminate 'T' from the cabin_df dataframe all together
cabin_df = cabin_df[cabin_df.Cabin != 'T']

# this will add two columns of a dataframe together. The existing columns 'SibSp' and 'Parch' are added to create new
# column called 'Alone'
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch

# you can change the values of a column in a dataframe in this way
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] ==0] = 'Alone'

# adding values of two columns using .loc
occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

# this will create a new column called 'Survivor', and its value will be the result of the .map function.
# this will take the current 'Survived' column and use its value and a dictionary to create a value in 
# the new column. If the value in the 'Survived' column is 1, the new column will read 'yes'. 
titanic_df['Survivor'] = titanic_df.Survived.map({1:'yes',0:'no'})

# this will calculate a percent change. A new column ('Daily Return') is added to AAPL dataset and is set equal
# to the percent change of the 'Adj Close' column.
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

# dropping specific columns from a dataframe by calling their exact names. avg is the dataframe. axis of 0
# is used to drop a row, axis of 1 is used to drop a column.
avg.drop(['Number of Observations','Question Text','Question Iteration'], axis=0, inplace=True)

# this will group the dataset by a certain column (same thing as group by in sql). you can then decide how to display
# the grouped information, and here the mean of the grouped values is shown.
poll_df = poll_df.groupby(['Start Date'], as_index=False).mean()

# this will createa dataframe out of an existing column from a dataset, but it will be a copy of the column.
# here, the dataset is 'donor_df' and the existing column is 'contb_receipt_amt'
top_donor = donor_df['contb_receipt_amt'].copy()

# sort the values of a dataset
top_donor.sort_values()

# Use a pivot table to extract and organize the data by the donor occupation
occupation_df = donor_df.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='Party', aggfunc='sum')

# This will take the column 'Inside/Outside' and find whereit equals 'Inside', then set it equal to 'I'
data[data['Inside/Outside'] == 'Inside'] = 'I'

# the above code caused problems so use this instead. this will replace any 'O' or 'I' the 'Weapon' column to 'X' 
data = data.replace({'inside/outside': ['O']}, 'Inside')

# to delete a variable or list, exact. here 'deck' is the variable
del deck

################################## FUNCTIONS #############################################################################

# this function will take in age,sex as parameters and return if the passenger was a m,f, or child
def male_female_child(passenger):
    age,sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

################################## LOOPS ################################################################################

# the following while loop will output only the letter in each cabin entry. first creating an empty list,
# for all entries (level) in deck, the first letter of the entry will be added to the list 'levels'
levels = []
for level in deck:
    levels.append(level[0])

# this is a loop that will determine all of the indexes which include October 2012 as the start date. We will plot
# the index range so we can see all poll resutls from October 2012. We can also do this by simply sorting the
# dataframe and manually checking the indexes, but this will not be useful for large dataframes or for instances
# in which specific rows must be found. So it is useful to create a loop to be able to do this for future use.
row_index = 0
xlimit = []
oct_day = []
for date in poll_df['Start Date']:
    if date[0:7] == '2012-10':
        xlimit.append(row_index)
        row_index += 1
        oct_day.append(date)
    else:
        row_index += 1
count = 0
for x in xlimit:
    print xlimit[count],oct_day[count]
    count += 1

# to print values in a particular way using a loop
cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()
i = 0
for don in cand_amount:
    print " The candidate %s raised %.0f dollars " %(cand_amount.index[i],don)
    print '\n'
    i += 1