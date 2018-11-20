#initiate
import plotly
plotly.tools.set_credentials_file(username = '', api_key = '')

import plotly.dashboard_objs as dashboard
import IPython.display
from IPython.display import Image
 
my_dboard = dashboard.Dashboard()
my_dboard.get_preview()





