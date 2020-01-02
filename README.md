# Ranking Under Lower Uncertainty

## Setup
This file assumes you have access to a *nix-like machine (both MacOS or
Linux would do).

The projects uses `pyenv` and `pipenv` for package management.
Before you start, please ensure you have `gcc`, `make`, and `pip` installed.

### Installing `pyenv`

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

### Installing `pipenv`
Install pipenv using `pip` (or `pip3`):
```
pip install -U pipenv
```

### Download the repository and sync the environment
```
git clone https://github.com/anonymous-authors1234/ranking_under_lower_uncertainty.git
cd ranking_under_lower_uncertainty

# Switch to Python 3.7.3 for pyenv
pyenv local 3.7.3
pipenv update --dev
```
