# Graph Matching Framework

This is a Cython framework to perform graph matching.
It uses the flexibility of python and the efficiency of compiled c code to 
package contains the core files to perform graph matching.

## Install

git clone https://github.com/CheshireCat12/graph-matching-core.git

Create python virtual environment with python3.9
python3.9 -m venv venv

source venv/bin/activate

pip install numpy

pip install -e .

Run tests:
pytest tests
