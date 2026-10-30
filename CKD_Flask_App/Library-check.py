import sys
import flask
import numpy as np
import pandas as pd
import sklearn
import pickle
from importlib.metadata import version
import gunicorn

def print_library_versions():
    print("=" * 50)
    print("Python Environment Information")
    print("=" * 50)
    print(f"Python   :{sys.version}")
    #print(f"Flask    :{flask.__version__}")
    print(f"Flask : {version('flask')}")
    print(f"Numpy    :{np.__version__}")
    print(f"Pandas   :{pd.__version__}")
    print(f"Scikit-learn  :{sklearn.__version__}")
    print("Gunicorn Version:", gunicorn.__version__)
    #gunicorn version is very important
    print("=" * 50)

print_library_versions()
    