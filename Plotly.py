#initiate
import plotly
plotly.tools.set_credentials_file(username = '', api_key = '')

import plotly.dashboard_objs as dashboard
import IPython.display
from IPython.display import Image
 
my_dboard = dashboard.Dashboard()
my_dboard.get_preview()



#plot value_counts()
import plotly.plotly as py
import plotly.graph_objs as go

cnt_PC = pd.DataFrame({'PComp': raw.ParentCompany.value_counts()})
cnt_PC = cnt_PC.sort_values(by = 'PComp', ascending = False)

data  = go.Data([
                go.Bar(
                x = cnt_PC.index,
                y = cnt_PC.PComp,
                marker = dict(
                        color = 'skyblue',
                        line = dict(
                        color = 'skyblue',
                        width = 1.5)
                        ),
#                 opacity = 0.6
        )])

layout = go.Layout(
    title = "Acquisitions by Top 7 Tech Companies"
    )

fig  = go.Figure(data = data, layout = layout)
url_1 = py.plot(fig, filename = 'bar-plot-cnt-PComp', auto_open = False)
py.iplot(fig, filename = 'bar-plot-cnt-PComp')


# cnt_PC = raw.ParentCompany.value_counts()
# cnt_PC.iplot(kind='bar', yTitle='Count', title='Acquisitions by Top 7 Tech Companies',
#              filename='bar-plot-cnt-PComp_2')








