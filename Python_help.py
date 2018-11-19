#import csv
train = pd.read_csv("directory")

#when csv text in spanish, not utf-8
df = pd.read_csv(file_path, encoding='latin-1')

#import pickle object
import pandas as pd
import pickle

df = pd.read_pickle('filePath.pkl')
df

#print dim of data
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print head of the dataset
print(train.head())

#check current working directory(in python or jupyter notebook)
import os
os.getcwd()

#change current working directory to path
os.chdir('path')



#check system path
import sys
sys.path


#check virtual env version in commandline
virtualenv --version

#create a virtual env
cd my_project_folder
virtualenv my_project

#activate virtual env
source my_project/bin/activate
 
pip install requests

###############    install packages in virtual env      ###############
pip install -U numpy scipy scikit-learn


#######################################################################



deactivate

#check python package version in commandline
Python
>>> import tensorflow
>>> print(tensorflow.__version__)


#outside of virtual environment
#jupyter path problem
python3 -m pip install --upgrade pip
python3 -m pip install jupyter


#import matplotlib error
#inside virtual env at terminal
cd ~/.matplotlib
nano matplotlibrc

#type
backend: TkAgg

#Crtl + O to save, Crtl + X to exit


#import seaborn error, import in terminal, not in jupyter notebook
#cause and solution: https://github.com/jupyter/notebook/issues/2359
#check in notebook

sys.executable
# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6


sys.path
# ['',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python36.zip',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/lib-dynload',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages',
#  '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/extensions',
#  '/Users/qinqingao/.ipython']



#reinstall seaborn in where sys.executable is
#in virtualenv
# sys.executable -m pip install seaborn

/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6 -m pip install --upgrade seaborn


#check GCC version
gcc --version

# Configured with: --prefix=/Applications/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
# Apple LLVM version 9.0.0 (clang-900.0.39.2)
# Target: x86_64-apple-darwin16.7.0
# Thread model: posix
# InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

#or

gcc -dumpversion | cut -f1,2,3 -d.

# 4.2.1
 
#change dataset shape from long to wide(pivot table)
from pandas import *
df = pd.DataFrame(dataset_name)
pt = pivot_table(df, values = 'value', index = ['well', 'tstamp'], columns = ['channel'], aggfunc = np.sum)
pt.head()

#export to csv
df.to_csv('path/name_of_file.csv', header = True, index = False)


#select subset of data based on condition in a column which starts with certain string
dataset2 = dataset1.loc[dataset1['colname'].str.startswith('abc')]

#select subset of data based on condition in a column which equals a certain string
dataset2 = dataset1.loc[dataset1['colname'] == 'abc']

#select subset of data based on location index
df.iloc[0:28, 16:40]


#find unique value of a column in a dataframe
df['colname'].unique()


#unique value of an array
set(arry)


#find unique, sorted, no NaN value of a column in a dataframe
uniq_sort_df = sorted(df['colname'].dropna().unique())

#find elements in a unique list if elements starts with string 'Abc'
[i for i in df['well'].unique() if i.startswith('Abc')]

#find data set whose column names start with 'Abc'
df[[i for i in df.columns.get_values() if i.startswith('Abc')]]


#find column names in df
list(df)

#widen pandas output
import pandas as pd
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

#summary() in R equivalent in Python
df.describe()

#count how many records in a column
len(df['colname'])

#select subset from different columns meet conditions, | means or
dataset1.loc[(dataset1['col1'] == 'abc') | 
             (dataset1['col2'] == 'def')]


#select subset of dataset1 from a list of specific values from a column in dataset1
dataset2 = dataset1[dataset1['col1'].isin(['abc', 'def'])]
dataset2


dataset1.loc[(dataset1['col1'] == 'abc') & 
             (dataset1['col2'] == 'def')]

#is equivalent to

dataset1[(dataset1['col1'] == 'abc') & 
         (dataset1['col2'] == 'def')]

#above return selected dataframe (subset of bigger dataframe dataset1)
          
#below will return the column needed 'col3' in that subset of dataframe
dataset1[(dataset1['col1'] == 'abc) & (dataset1['col2'] == 'def')]['col3']

          
#below will return the first cell in column 'col3' in the subset of dataframe
dataset1[(dataset1['col1'] == 'abc) & (dataset1['col2'] == 'def')]['col3'].iloc[0]

          
          
#python index of these rows whose columns meet certain conditions
dataset2 = dataset1.loc[(dataset1['col1'] == 'abc') | 
                        (dataset1['col2'] == 'def')]

drop_index = dataset2.index.tolist()

#drop rows from data frame based on row index
dataset1.drop(dataset1.index[drop_index])


#drop some records/rows meet condtion
df = df.drop(df[pd.to_datetime(df['ts']) >= dict[key]].index)
          
          
#check if a column is sorted(without NA)
(sorted(df.colname.dropna()) == df.colname.dropna()).unique()


#select columns in df by column name
df.loc[:, df.columns.isin(['colNameA', 'colNameB'])]

          
#select columns in df by column name, return first 30 rows          
df.loc[ : , ['colA', 'colB']][df['colC'] == 1][:30]          

         
#make a table of counts, sort descendingly
lista.value_counts().sort_values(ascending = False)

          
#split column into separate columns by .       
df['a'] = df['a.b.c'].str.rsplit('.').str[1]
df['b'] = df['a.b.c'].str.rsplit('.').str[2]
df['c'] = df['a.b.c'].str.rsplit('.').str[3]          

          
#space out time with dates and frequency specified
pd.date_range('2013-08-01 00:00:00', '2017-03-06 00:00:00', freq='20min')


#cartesian product, level 2 (longer list) map to level1
index = pd.MultiIndex.from_product([lista, listb], names = ['cola', 'colb'])
pd.DataFrame(index = index).reset_index()

	  
#rename index column	  
df.index.name = 'new_col_name'	  
	  

#assign groups based on conditions
conditions = [df['colname'] == 0, 
              df['colname'] <= 20,
              df['colname'] <= 40,
              df['colname'] < 60]
choices = ['P3', 'P1', 'P2', 'P3']
df['colname'] = np.select(conditions, choices)


#create new column New_Col according to conditions in other columns          
def fct(row):
    if (row['ColA'] == 1) and (row['ColB'] == 'Abc'):
        return 'Big'
    elif (row['ColA'] == 1) and (row['ColB'] == 'Def'):
        return 'Medium'
    elif row['ColA'] == 0:
        return 'Small'

df['New_Col'] = df.apply(lambda row: fct(row), axis = 1)          
 
          
          
#convert datatype to str for columns in dataframe
df['new_col'] = df['colA'].astype(str) + ' '+ df['colB'].astype(str) + ':' + df['colC'].astype(str) + ':00'
df


#convert datatype from str to datetime
df['new_colName'] = pd.to_datetime(df['colName'])


#left join, SQL-like
df_new = pd.merge(df_ontheleft, df_ontheright, how = 'left', on = ['commonColA', 'commonColB'])

# add a Flag column based on whether another column's value is null or not
X['Flag'] = np.where(pd.notna(X['Date']), 1, 0)          
          
          
#check number of NAs in each column of a dataframe
df.isnull().sum()

          
#return dataframe where colA has NaN
df[df.ColA.isnull()]
          

#calculate time delta between two timestamps
(pd.to_datetime(max(df_fillin_interpolate['tstamp'])) - pd.to_datetime(min(df_fillin_interpolate['tstamp'])))


#find out rows where a column doesn't have nulls
df[df['name'].notnull()]

#number of rows where a column doens't have nulls
len(df[df['name'].notnull()])


#interpolate missing value only if the gap is not too big
https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe
 
mask = data.copy()
grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
grp['ones'] = 1
for i in list('abcdefgh'):
    mask[i] = (grp.groupby(i)['ones'].transform('count') < 5) | data[i].notnull()
  
  
#interpolate all NaNs in df_a first, then back fill with the ones that need to be NaNs
df = df_a.interpolate().bfill()

#backfill according to the True False table that mask has, exclude first 3 columns
df_interpolate_w_rule = df.iloc[:, 3:][mask == True]

#put back first 3 columns together with channel value and finish the imputation, call the dataframe df_imputed
first_3_col = df_a.iloc[:, 0:3]

#put two dataframes together, side by side
df_imputed = pd.concat([first_3_col, df_interpolate_w_rule], axis = 1)

          
#concatenate dataframe and dictionary side by side, handle index mismatching          
df_c = pd.DataFrame()
df_results = pd.DataFrame()

for item in item_list:
    df_c = pd.concat([df[df['colA'] == item].reset_index(), pd.DataFrame(dict[item]).reset_index()], axis = 1)
    #indent here is important!
    df_results = df_results.append(df_c)[df_c.columns.tolist()]
  
          
#To keep columns same order in the appended dataframe
df_full = df_full.append(df)[df.columns.tolist()]

          
#describe statistics of a set of columns whose names start with certain string
df[[i for i in df.columns.get_values() if i.startswith('Abc')]].dropna().describe()

#select several index columns from the same data frame
df[['a', 'b']]

          
#select colA, D, F by name from df and make it a new dataframe: df_new          
df_new = df.loc[:, ['colA', 'colD', 'colF']]          
          
          
#create a new column that flags 1 when any of the two columns is 1
df['flag'] = df[['a', 'b']].max(axis = 1)

#convert time duration from days to hours
df['time_down'].dt.total_seconds() / 3600

#extract part of string for all strings in a list
[i[-4 : -1] for i in listA]

#drop duplicates in both columns
df[['colA', 'colB']].drop_duplicates()

#remove dataframe need to be deleted from df1
#repeat df2 because it's not a complete subset of df1
#our goal is to negate those from df2 in df1, not to include what's in second one

pd.concat([df1, df2, df2]).drop_duplicates(keep = False)

#feature engineering
for i in list(df):
    for j in [3, 24]:
        #moving average, standard deviation, coefficient of variation, median, 50% quantile
        #for time window of 3, 24 hours, or window = 9, 72 intervals (1 interval = 20 min)
        #min_periods (min number of observations in window before returning NaN) is set to be > half of data available in the window
        df[i + '_mean' + str(j)] = df[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).mean()
        df[i + '_sd' + str(j)] = df[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).std()
        df[i + '_cov' + str(j)] = df[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).std()/df_imputed[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).mean()
        df[i + '_median' + str(j)] = df[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).median()
        df[i + '_50pct' + str(j)] = df[i].rolling(window = 3*j, min_periods = math.ceil(3*j/2)).quantile(.5)
        df[i + '_mean1diff' + str(j)] = df[i].diff().rolling(window = 3*j - 1, min_periods = math.ceil(3*j/2)).mean()

          
##################################### matplotlib #####################################

#plot use ggplot style
plt.style.use('ggplot')
          
          
#plot y1, y2, y3 wrt x, change line style to None, and marker as dots (look like scatter plot)
plt.plot(x, y1, x, y2, x, y3, linestyle = 'None', marker = 'o')

fig = plt.gcf()

#format the ticks, cutomize N
fig.autofmt_xdate()
xmin, xmax = plt.gca().get_xlim()
plt.gca().set_xticks(np.round(np.linspace(xmin, xmax, N), 2))

#add vertical lines
xcoords = [1, 2]
for xc in xcoords:
    plt.axvline(x=xc)

          
#plot multiple vertical lines stored in a list
[plt.axvline(_x, color = 'red', linestyle = '--') for _x in ListA[i].astype(str)]          

          
#make legend
fig.legend(['A', 'B'])

#increase figure size
fig.set_size_inches(25, 15)

#add title for y axis, figure
plot_title = 'Abc'
plot_ylabel = 'Temperature (F)'

plt.title(str(plot_title), fontsize = 35)
plt.ylabel(plot_ylabel)


#custom size
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize


fig.savefig(filepath + str(plot_title) + '.png')      
       
          
          
################################ Size: title, x,y label, x,y ticks ################################
          
plt.xlabel('Age', fontsize = 16)
plt.ylabel('Distance (mile)', fontsize = 16)
plt.title('Cluster of Millenials Regular Employees, k = 4', fontsize = 22, fontweight = "bold")

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)          
 
####################################################################################################          
          
#loop, print
#plot delta pressure figures for when pumps fail

well_name = ['a', 'b', 'c', 'd', 'e']

#custom size
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size = MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = MEDIUM_SIZE)    # legend fontsize

#following lists need to be same length, for each index of each list, column needs to be same length
x = [col1['tstamp'], col2['tstamp'], col3['tstamp'], col4['tstamp'], col5['tstamp']]
y1 = [col1['abc'], col2['abc'], col3['abc'], col4['abc'], col5['abc']]
y2 = [col6['def'], col7['def'], col8['def'], col9['def'], col10['def']]


for i in range(len(x)):
    plt.figure()
    # plot
    fig = plt.plot(x[i], y1[i] - y2[i], linestyle = 'None', marker = 'o')

    fig = plt.gcf()

    # format the ticks
    fig.autofmt_xdate()
    xmin, xmax = plt.gca().get_xlim()
    plt.gca().set_xticks(np.round(np.linspace(xmin, xmax, 12), 2))


    #make legend
    fig.legend(['ghi'])


    #increase figure size
    fig.set_size_inches(25, 15)


    #add title for y axis, figure
    plot_ylabel = 'ijk)'
    plot_title = 'Failure for ' + str(well_name[i]) + ' wrt ' + str(plot_ylabel)


    plt.title(str(plot_title), fontsize = 35)
    plt.ylabel(plot_ylabel)
    
#     fig.savefig(filepath + str(plot_title) + '.png')
          

#check column type
colA.dtype

          
#find timestamp that is a days before tstart
plot_start = tstart - timedelta(days = a)
print(plot_start)

#find timestamp that is b days after tend
plot_end = tend + timedelta(days = b)
print(plot_end)          
          
          
#find all timezones available in Python
import pytz
pytz.all_timezones
          

          
#https://matplotlib.org/gallery/subplots_axes_and_figures/ganged_plots.html          
fig = plt.figure()
# set height ratios for sublots
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
line0, = ax0.plot(x1[1], y1[1])

ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(x2[1], y3[1])
plt.setp(ax0.get_xticklabels(), visible = False)

##remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()

# https://stackoverflow.com/questions/42973223/how-share-x-axis-of-two-subplots-after-they-are-created

          
t= [[22,23],[24,25],[26,27]]
x = [[10,20],[30,40],[50,60]]
y = [[30,40],[50,60],[70,80]]


for i in range(len(x)):
    fig=plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    ax1.plot(t[i], x[i])
    ax2.plot(t[i], y[i])
    
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])
    ax1.set_title('Set ' + str(i))
    # ax2.autoscale() ## call autoscale if needed
plt.show()
          

          
for i in range(len(x1)):
    fig = plt.figure(figsize = (25, 55))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413)
    ax4 = plt.subplot(414)


    # plot, and add vertical line to signal start of failed event for each subplot
    ax1.plot(x1[i][1:100], y1[i][1:100], x1[i][1:100], y2[i][1:100], linestyle = 'None', marker = 'o')
    [ax1.axvline(_x, color = 'red', linestyle = '--') for _x in fail_start_gmt[i].astype(str)]
    [ax1.axvline(_x, color = 'green', linestyle = '-') for _x in fail_end_gmt[i].astype(str)] 
    
    ax2.plot(x2[i][1:100], y3[i][1:100] - y4[i][1:100], linestyle = 'None', marker = 'o')  
    [ax2.axvline(_x, color = 'red', linestyle = '--') for _x in fail_start_gmt[i].astype(str)]
    [ax2.axvline(_x, color = 'green', linestyle = '-') for _x in fail_end_gmt[i].astype(str)]    
    
    ax3.plot(x3[i][1:100], y5[i][1:100], x3[i][1:100], y6[i][1:100], linestyle = 'None', marker = 'o')
    [ax3.axvline(_x, color = 'red', linestyle = '--') for _x in fail_start_gmt[i].astype(str)]
    [ax3.axvline(_x, color = 'green', linestyle = '-') for _x in fail_end_gmt[i].astype(str)]    
    
    ax4.plot(x4[i][1:100], y7[i][1:100], linestyle = 'None', marker = 'o')
    [ax4.axvline(_x, color = 'red', linestyle = '--') for _x in fail_start_gmt[i].astype(str)]
    [ax4.axvline(_x, color = 'green', linestyle = '-') for _x in fail_end_gmt[i].astype(str)]    
    
    
#     ax1.get_shared_x_axes().join(ax1, ax2, ax3, ax4)
  
    
    # plot merged figure title
    ax1.set_title('Failure for ' + str(listA[i]), fontsize = 30)
    
    # show legend for each subplot
    ax1.legend(['abc', 'def'], loc = 1)
    ax2.legend(['cdc'], loc = 1)
    ax3.legend(['dfg', 'efg'], loc = 1)
    ax4.legend(['fge'], loc = 1)
       
    
    # show individual y label for each subplot
    ax1.set(ylabel = 'Temperature (F)')
    ax2.set(ylabel = 'Pressure (psi)')
    ax3.set(ylabel = 'Voltage (V)')
    ax4.set(ylabel = 'Vibration (gn)')
               
          
    # format the x ticks to 12 intervals, for easy read
    fig = plt.gcf()
    fig.autofmt_xdate()
    xmin, xmax = plt.gca().get_xlim()
    plt.gca().set_xticks(np.round(np.linspace(xmin, xmax, 12), 2))


    #elimiate vertical space (height) in between subplots    
    plt.subplots_adjust(hspace = 0)
    
plt.show()          

          
#plot correlation heatmap using seaborn
fig = plt.figure(figsize = (10, 8))
corr = data.corr()
sns.set(font_scale = 1.6)
sns.heatmap(corr, annot = True, cmap = 'seismic')
plt.title('Heatmap of Correlation Matrix', fontsize = 25)
          
          
          
          
#partition big dataset with smaller dataset based on unique value in a column          
dict_of_items = {item: df_item for item, df_item in df.groupby('Items')

# access first item of first value of key from a dictionary
list(dict['key1'][0])[0]

                 
#merge several csvs into 1 csv
import pandas as pd
import glob, os
 
os.chdir('filepath')
results = pd.DataFrame([])
 
for counter, file in enumerate(glob.glob("csv_filename_matching*")):
    namedf = pd.read_csv(file)
    results = results.append(namedf)
 
results.to_csv('path', header = True, index = False)
                 

#get first 80% of data from files in this directory that start with specific string

for counter, file in enumerate(glob.glob('csv_filename_matching*')):
    namedf = pd.read_csv(file)
    results = results.append(namedf[: round(0.8 * (namedf.shape[0]))])
                 
                               
                
#check and return columns names whose value is all null
results.columns[results.isnull().all()]                 
                 
#find columns that have all nulls, and drop those columns by column names
#before dropping those all null columns, check dataframe shape
results.shape
                 
nullcols = results.columns[results.isnull().all()]
print(nullcols)
print(len(nullcols))
                 
results.drop(nullcols, inplace = True, axis = 1)
                             
#after dropping those all null columns, check dataframe shape                 
results.shape
                 
list(results)
                 
      
#time how long a step in ipython notebook takes
 %time (code)
                 
                 
#make a dictionary based on value in a list
                 
dict = {}
                 
for item in i_list:
    try:
        dict['{0}'.format(item)] = dataframe/value/tuples
    except (SyntaxError, KeyError):
        pass
 
                 
#dataframe
                 
dict = {}

for item in i_list:
    try:
        dict['{0}'.format(well)] = df[(pd.to_datetime(df['colD']) >= dict_a[j]) & (pd.to_datetime(df['colD']) <= dict_b[j]) & (df['col_item'] == i)]  
    except (SyntaxError, KeyError):
        pass                 
                 
                 
#value
                          
dict = {}

for item in i_list:
    try:
        dict['{0}'.format(well)] = pd.to_datetime(df[(df['colC'] == 1) & (df['col_item'] == i)]['colA'].iloc[0])                   
    except (IndexError):
        pass
print(dict)                 
                 
                 
                 
#tuple                 
dict = {}

for item in i_list:
    try:
        dict['{0}'.format(i)] = (pd.to_datetime(df[(df['colC'] == 1) & (df['col_item'] == i)]['ColA'].iloc[0]) - timedelta(days = 10), pd.to_datetime(df[(df['colC'] == 1) & (df['col_item'] == i)]['colB'].iloc[-1]) + timedelta(days = 10))
    except (IndexError):
        pass
print(dict)               
                 

                 
#find out for each, what end time does it have

dict_t = {}
tot = 0

for item in item_list:
    try:
        dict_t['{0}'.format(item)] = pd.to_datetime(df_log[(df_log['flag'] == 1) & (df_log['Col_item'] == item)]['ts'])           
        print(item)
        print(len(list(dict_t[item])))
        tot += len(list(dict_t[item]))
        
    except (IndexError):
        pass

print(tot)
print(dict_t)
                 
                 
                 
                 
#rename column name                 
df.rename(columns = {'oldNameColA': 'newNameColA', 'oldNameColB': 'newNameColB'}, inplace = True)   

                 
#find rows where colA is null in df                 
df[df['colA'].isnull()] 
                 
                 
#update partial column value to 1 based on another column (ColB == 1) in the df                 
#notice no quotation mark on ColB
                 
df['col_target'][(df.ColB == 1)] = 0                 
                 

#col value based on if-else in another column
df['col_target'] = np.where(df['ColB'] == 1, 'yes', 'no')
                 
                 
#count number of 1s in df changed-to list
list(df.loc[ : , 'colFlag'][(df['colFlag'] == 1) & (df['colA'] == 'Abc')]
).count(1)

                 
#if any number in the series is greater than a threshold, then increase count by 1

pred = 0
if any(i >= thres for i in rmse):
    pred += 1
       

#convert list to string                 
''.join(df['Abc'].unique())


                 
                 
#create a dictionary, if a key already exist, append value to it; if not, create new key, and corresponding value                 
dict_full = {}

                 
import glob, os
os.chdir(path)
                 
for counter, file in enumerate(glob.glob('startWithAbc_*')):
    df = pd.read_csv(file) 
    try:
        dict_full[''.join(df['Abc'].unique())].append(df)
    except (KeyError):
        dict_full[''.join(df['Abc'].unique())] = [df]          

                 
#if column Abc has one single value                 
dict_a = {}

for counter, file in enumerate(glob.glob('dict_*')):
    namedf = pd.read_csv(file)
    print(namedf.well[0])
    try:
        dict_a[namedf.Abc[0]].append(namedf)
        
    except (KeyError):
        dict_a[namedf.Abc[0]] = [namedf]
                 
       
                 
#loop 2 variables simultaneously
for i, j in zip(glob.glob('dict_*'), j_list):
    dict_a[j][0]['new_col'] = dict_b[j] - pd.to_datetime(dict_a[j][0]['col_ts'])
    print(dict_a[j])
    dict_a[j][0].to_csv(path + str(file[11 : -4]) + '.csv', header = True, index = False)

                 
                 
#install survival model, datasets, which is on top of scikit-learn
sys.executable -m pip install --upgrade scikit-survival
             
                 
#combine year, month, day, hour to 1 timestamp column when importing        
from pandas import read_csv
from datetime import datetime
                 
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H') 
                 
dataset = read_csv(path, parse_dates = [['year', 'month', 'day', 'hour']], index_col = 0, date_parser = parse)
dataset.head()                 
                 
                 
                 
#replace missing value NaN in column Abc with 0                
dataset['Abc'].fillna(0, inplace = True)


                 
#plot pred vs actual                 
plt.figure(figsize = (10, 6))
plt.plot(yhat[ : 1000])
plt.plot(y[ : 1000])
plt.legend(['Predict', 'Actual'], loc = 'upper right')
plt.ylim(0, 20)
plt.show()                 
                 

#LSTM architecture
nb_epoch = 100
batch_size = 256

model = Sequential()

#hidden layer with 20 neurons
model.add(LSTM(20, input_shape = (train_X.shape[1], train_X.shape[2]), activation = 'relu'))

#ouput layer has 1 single value for prediction
model.add(Dense(1)) 

model.compile(loss = 'mae', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])


#Set callback functions to early stop, and save the best model so far
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5),
             ModelCheckpoint(filepath = 'best_model.h5', monitor = 'val_loss', save_best_only = True)]

#fit network
history = model.fit(train_X, train_y, 
                    epochs = nb_epoch, 
                    batch_size = batch_size, 
                    validation_data = (test_X, test_y), 
                    verbose = 1, 
                    shuffle = False,
                    callbacks = callbacks)
                 
                 
#Geocode
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode = True)
                 
home_zip = search.by_zipcode("08648")
print(home_zip)
lat_long = (home_zip.lat, home_zip.lng)
print('\n')
print('Latitude and Longitude are: ' + str(lat_long))
                 
#Create tuple of latitude, longitude, 'Lat_Lng_Zip' based on zip code column
df['Lat_Lng_Zip'] = df.apply(lambda df: (search.by_zipcode(df['Zip Code']).lat, search.by_zipcode(df['Zip Code']).lng), axis = 1)

                 
                 
#another zipcode package
from pprint import pprint
import zipcodes
                 
def homeZip_to_geocode(zip):
    try:
        return (zipcodes.matching(str(zip))[0]['lat'], zipcodes.matching(str(zip))[0]['long'])
    except:                       
        pass

X['Lat_Lng_Home2'] = X['Zip Code'].apply(homeZip_to_geocode)

                 
                 
                 
# Create Distance_mile to calculate home to office distance in miles                 
from geopy.distance import vincenty

def distance_calc(row):
    try:
        start = row['Lat_Lng_Home']
        stop = row['Lat_Lng_Office']
        return vincenty(start, stop).miles
    except:
        pass                 

X['Distance_mile'] = X.apply (lambda row: distance_calc(row), axis = 1)         
                 
                 
#Create a map from a column, and generate a new column with mapped value
my_map = {'Abc' : '1', 'Def' : '2', 
                  'Ghi' : '3', 'Jkl' : '4'}
                 
df['New Col'] = df['Old Col'].map(my_map)                 
                 
               
#plot density function of Distance_mile for different groups
plt.figure(figsize = (14, 8))

ax1 = sns.kdeplot(X_mill_r_qfs.Distance_mile[(X_mill_r_qfs['colA'] == 1) & (X_mill_r_qfs['colB'] == 'Abc')], label = 'Big', color = 'b', shade = True)
ax2 = sns.kdeplot(X_mill_r_qfs.Distance_mile[(X_mill_r_qfs['colA'] == 1) & (X_mill_r_qfs['colB'] == 'Def')], label = 'Medium', color = 'r', shade = True)
ax3 = sns.kdeplot(X_mill_r_qfs.Distance_mile[X_mill_r_qfs['colA'] == 0], label = 'Small', color = 'g', shade = True)

plt.xlabel('Distance (miles)', fontsize = 16)
plt.ylabel('Density', fontsize = 16)
plt.title('XYZ', fontsize = 22, fontweight = "bold")

plt.setp(ax1.get_legend().get_texts(), fontsize = 14)
                 
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

ax1.set_xlim(0, 500)
ax2.set_xlim(0, 500)
ax3.set_xlim(0, 500)

plt.show()

                 
                 
#survival by groups                 
model_grp <- survfit(Surv(as.numeric(Length_Stay), Term.1..Stay.0.) ~ Group, data = train)

grp_plot <- autoplot(model_grp) + labs(x = 'Time', y = 'Survival Probability (%)', title = 'XYZ') +
				  theme(plot.title = element_text(hjust = 0.5), title = element_text(size = 14, face = 'bold'),
					axis.text.x = element_text(face = 'bold', size = 12),
					axis.text.y = element_text(face = 'bold', size = 12))
cols <- c('1' = 'green', '0' = 'red')
grp_plot_label <- grp_plot +  guides(fill = FALSE) + labs(color = 'Group') + 
		  scale_color_manual(values = cols, breaks = c('1', '0'), labels = c('Grp A', 'Grp B'))
print(grp_plot_label)                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
