import pandas as pd
import numpy as np
import setup_env as env
from sklearn.utils import shuffle
import xgboost
import pickle
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "AUC", "a", "b", "TrueNeg", "FalsePos", "FalseNeg", "TruePos" ])


df = pd.read_sql(env.str_sql_data, env.engine_sf_tandr)