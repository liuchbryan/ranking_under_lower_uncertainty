# Ranking Under Lower Uncertainty

The repository contains the code and supplementary documents associated 
with the paper [What is the value of experimentation and measurement?](https://link.springer.com/article/10.1007%2Fs41019-020-00121-5), 
which has been published in the Data Science and Engineering Journal. 
An earlier version of the paper appeared in the [IEEE ICDM 2019 conference](https://ieeexplore.ieee.org/document/8970749), which
used [a separate codebase](https://github.com/liuchbryan/value_of_experimentation).

If you find the code useful for your work, please consider citing the underlying article

```bibtex
@article{liu2020valueofexpt,
    author={Liu, C. H. Bryan and Chamberlain, Benjamin Paul and McCoy, Emma J.},
    title={What is the Value of Experimentation and Measurement?},
    journal={Data Science and Engineering},
    year={2020},
    month={Jun},
    day={01},
    volume={5},
    number={2},
    pages={152-167},
    issn={2364-1541},
    doi={10.1007/s41019-020-00121-5},
    url={https://doi.org/10.1007/s41019-020-00121-5}
}
```

# Setup
This file assumes you have access to a *nix-like machine (both MacOS or
Linux would do).

The projects uses `pyenv` and `pipenv` for package management.
Before you start, please ensure you have `gcc`, `make`, and `pip` installed.

## Installing `pyenv`

For Linux (together with other required libraries):

``` bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
wget -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

chmod u+x pyenv-installer
./pyenv-installer
```

For OS X:
```
brew install pyenv
```

We then need to configure the PATHs:
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

...and install the right Python version for our environment:
```
pyenv install 3.7.3
```

### Installing `poetry`
See https://python-poetry.org/docs/#installation for the installation instructions.

### Download the repository and sync the environment
```
git clone https://github.com/liuchbryan/ranking_under_lower_uncertainty.git
cd ranking_under_lower_uncertainty


# Switch to Python 3.7.3 for pyenv
pyenv local 3.7.3
poetry install
```

### Run the Jupyter notebooks  
```
poetry shell
```

Within the newly spawn up virtualenv shell, run
```
jupyter notebook
```

Once you are done, terminate the Jupyter server using Ctrl+C, and type `exit` to exit the virtualenv shell.  
 

