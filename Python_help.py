#import csv
train = pd.read_csv("directory")

#when csv text in spanish, not utf-8
df = pd.read_csv(file_path, encoding='latin-1')


#print dim of data
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

#print head of the dataset
print(train.head())

#check current working directory(in python or jupyter notebook)
import os
os.getcwd()

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
# check in notebook

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
df.to_csv(r'.\raw_data_shushu_nodup.csv', header = True, index = False)


#select subset of data based on condition in a column which starts with certain string
dataset2 = dataset1.loc[dataset1['colname'].str.startswith('abc')]

#select subset of data based on condition in a column which equals a certain string
dataset2 = dataset1.loc[dataset1['colname'] == 'abc']

#select subset of data based on location index
df.iloc[0:28, 16:40]


#find unique value of a column in a dataframe
df['colname'].unique()

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



dataset1.loc[(dataset1['col1'] == 'abc') & 
            (dataset1['col2'] == 'def')]

# is equivalent to

dataset1[(dataset1['col1'] == 'abc) & 
          (dataset1['col2'] == 'def')]

# above return selected dataframe (subset of bigger dataframe dataset1)
          
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


#check if a column is sorted(without NA)
(sorted(df.colname.dropna()) == df.colname.dropna()).unique()


#make a table of counts, sort descendingly
lista.value_counts().sort_values(ascending = False)


#space out time with dates and frequency specified
pd.date_range('2013-08-01 00:00:00', '2017-03-06 00:00:00', freq='20min')


#cartesian product, level 2 (longer list) map to level1
index = pd.MultiIndex.from_product([lista, listb], names = ['cola', 'colb'])
pd.DataFrame(index = index).reset_index()


#assign groups based on conditions
conditions = [df['colname'] == 0, 
              df['colname'] <= 20,
              df['colname'] <= 40,
              df['colname'] < 60]
choices = ['P3', 'P1', 'P2', 'P3']
df['colname'] = np.select(conditions, choices)


#convert datatype to str for columns in dataframe
df['new_col'] = df['colA'].astype(str) + ' '+ df['colB'].astype(str) + ':' + df['colC'].astype(str) + ':00'
df


#convert datatype from str to datetime
df['new_colName'] = pd.to_datetime(df['colName'])


#left join, SQL-like
df_new = pd.merge(df_ontheleft, df_ontheright, how = 'left', on = ['commonColA', 'commonColB'])


#check number of NAs in each column of a dataframe
df.isnull().sum()


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

#describe statistics of a set of columns whose names start with certain string
df[[i for i in df.columns.get_values() if i.startswith('Abc')]].dropna().describe()

#select several columns from the same data frame
df[['a', 'b']]

#create a new column that flags 1 when any of the two columns is 1
df['flag'] = df[['a', 'b']].max(axis = 1)

#convert time duration from days to hours
df['time_down'].dt.total_seconds() / 3600

#extract part of string for all strings in a list
[i[-4 : -1] for i in listA]

#drop duplicates in both columns
df[['colA', 'colB']].drop_duplicates()


##################################### matplotlib #####################################


#plot y1, y2, y3 wrt x, change line style to None, and marker as dots (look like scatter plot)
plt.plot(x, y1, x, y2, x, y3, linestyle = 'None', marker = 'o')

fig = plt.gcf()

# format the ticks, cutomize N
fig.autofmt_xdate()
xmin, xmax = plt.gca().get_xlim()
plt.gca().set_xticks(np.round(np.linspace(xmin, xmax, N), 2))

# add vertical lines
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

fig, axs = plt.subplots(2, 1, sharex = True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace = 0)

# Plot each graph
axs[0].plot(x1[1], y1[1])
axs[1].plot(x2[1], y3[1])

plt.show()


  
          
t= np.arange(1000)/100.
x = np.sin(2*np.pi*10*t)
y = np.cos(2*np.pi*10*t)

fig=plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(t,x)
ax2.plot(t,y)

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])
# ax2.autoscale() ## call autoscale if needed

plt.show()
          
          
       

          
fig = plt.figure()
# set height ratios for sublots
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
line0, = ax0.plot(x1[1], y1[1])

ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(x2[1], y3[1])
plt.setp(ax0.get_xticklabels(), visible = False)

# # remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()

# https://stackoverflow.com/questions/42973223/how-share-x-axis-of-two-subplots-after-they-are-created
