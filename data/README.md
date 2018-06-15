# Data

The files located in the [raw](./raw) directory are the files 
that result from pulling the data files directly from 
the UCI [urls](./urls.txt) using `wget`. 

The files located in [clean](./clean) directory are the files after 
running the [format_data.py](../src/format_data.py) script on them.
Essentially all this is doing is moving the variable we are predicting 
into the first column and dropping any unecessary columns.  

### Column names

Auto-MPG dataset info from 
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names).

Housing dataset info from 
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names).

Yacht dataset info from 
[here](http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics).

### TODO: handle the stupid formatting in auto-mpg. 