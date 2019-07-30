# Python_coding

* mac: "sys.executable" -m pip install --upgrade pkg
* ensure package also runs in Jupyter notebook(kernel also inside the same virtual environment, other than in python):
    * (pip install ipykernel)
    * **ipython kernel install —-user —-name=custom_virenv_name**
    * go to jupyter (top right corner), choose this kernel named after custom_virenv_name
* windows: pip install --target=path pkg
* windows numpy installation problem: conda install -c anaconda numpy
* when enconter module not found error in jupyter but module can be imported through commandline on windows:  
  * in command line, install pkg; python, import sys, sys.path, copy the last one that has 'site-packages'  
  * in jupyter, import sys, sys.path, sys.path.append("path\site-packages")  
* show all virtual environments available: conda info --envs
* create new virtual environment using conda: conda create -n custom_virenv_name
* install pkgs with conda: conda install -n custom_virenv_name pkg_name
* class variables are variables being shared with all instances(objects)





