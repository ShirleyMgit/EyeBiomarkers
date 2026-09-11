import pandas as pd

if int(pd.__version__.split(".")[0]) < 3:
    pd.options.mode.copy_on_write = True
