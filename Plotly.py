#initiate
import plotly
plotly.tools.set_credentials_file(username = '', api_key = '')

import plotly.dashboard_objs as dashboard
import IPython.display
from IPython.display import Image
 
my_dboard = dashboard.Dashboard()
my_dboard.get_preview()



#plot value_counts() in bar chart
import plotly.plotly as py
import plotly.graph_objs as go

cnt_PC = pd.DataFrame({'PComp': raw.ParentCompany.value_counts()})
cnt_PC = cnt_PC.sort_values(by = 'PComp', ascending = False)

data  = go.Data([
                go.Bar(
                x = cnt_PC.index,
                y = cnt_PC.PComp,
                text = cnt_PC.PComp,
                textposition = 'outside',
                marker = dict(
                        color = 'orange',
                        line = dict(
                        color = 'orange',
                        width = 1.5)
                        ),
                opacity = 0.6
        )])

layout = go.Layout(
    width = 1000,
    height = 600,
    title = "<b>Acquisitions by Top 7 Tech Companies</b>",
    yaxis = go.layout.YAxis(
    title = 'Count'
    ))

fig  = go.Figure(data = data, layout = layout)
url_1 = py.plot(fig, filename = 'bar-plot-cnt-PComp', auto_open = False)
py.iplot(fig, filename = 'bar-plot-cnt-PComp')


# cnt_PC = raw.ParentCompany.value_counts()
# cnt_PC.iplot(kind='bar', yTitle='Count', title='Acquisitions by Top 7 Tech Companies',
#              filename='bar-plot-cnt-PComp_2')





#plot donut chart
amt_PC = pd.DataFrame(raw.groupby(['ParentCompany'])['Value (USD)'].sum())
amt_PC = amt_PC.sort_values(by = 'Value (USD)', ascending = True)

fig = {
  "data": [
    {
      "values": amt_PC['Value (USD)'],
      "labels": amt_PC.index,
      "domain": {"x": [0, 1]},
      "hoverinfo":"label",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"<b>Amount of Acquisition (USD) by Top 7 Tech Companies</b>",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Top 7",
                "x": 0.5,
                "y": 0.5
            }
        ]
    }
}


url_2 = py.plot(fig, filename = 'donut', auto_open = False)
py.iplot(fig, filename = 'donut')





























