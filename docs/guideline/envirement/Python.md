# Setup your local enviremtnt

## Install python if you do not have it:
- I recommend [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for easy pythhon version mangement. In this way you can easily install and switch between ptyhon versions. 
- [anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/overview) is really good alternative which comes with preinstalled conda and python pacakges and ready for work envirement out of the box.
-  [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)


In my setup I use pyenv `2.6.22-2-gb2a43bbc` version on Ubuntu 24.04
``` bash
curl -fsSL https://pyenv.run | bash
```
Now you need to install python version:
``` bash 
pyenv install 3.14.3
```
You always can view all avaiable python versions to install:
```bash
 pyenv install --list
```
Set you local python version 
```bash
pyenv local 3.14.3
```

It is good practice to create virtual envirement per project (not to use global)
``` bash
python -m venv venv
```

Now we should activate virtual envirement
``` bash 
source venv/bin/activate
```

And finally now we can install python packages via pip or directly form requretments.txt

``` bash 
pip install numpy
```

``` bash 
pip install -r requiremetns.txt
```