import pandas as pd
import numpy as np
import feather

def numpy2feather(ndarray, columns):
    df = pd.DataFrame(ndarray)
