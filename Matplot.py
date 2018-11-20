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





