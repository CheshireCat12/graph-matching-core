# Graph Matching Framework

This is a Cython framework to perform graph matching.
It uses the flexibility of python and the efficiency of compiled c code to 
package contains the core files to perform graph matching.

## Installation

#### Prerequisites
- Python 3.9
- Numpy
- A C++ compatible compiler


#### Install
```bash
# Clone repo
git clone https://github.com/CheshireCat12/graph-matching-core.git [folder-name]
# Move to repo directory
cd [folder-name]

#Create python virtual environment with python
python -m venv venv
source venv/bin/activate

# Install Numpy
pip install numpy

# Compile and install the code
pip install -e .
```

#### Run tests
```bash
python -m pytest tests
```
