import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Union


CORR_METHODS = ["pearson", "spearman"]


class WeightedCorr:
    def __init__(self,
                 xyw: pd.DataFrame = None,
                 x: pd.Series = None, y: pd.Series = None, w: pd.Series = None,
                 df: pd.DataFrame = None, wcol: str = None):
        """Weighted Correlation class.
        To instantiate WeightedCorr, either supply:
          1. xyw as pd.DataFrame,
          2. 3 pd.Series: (x, y, w),
          3. a pd.DataFrame and the name of the weight column.
        Args:
          xyw: pd.DataFrame with shape(n,3) containing x, y, and w columns;
          x: pd.Series (n,) containing values for x;
          y: pd.Series (n,) containing values for y;
          w: pd.Series (n,) containing weights;
          df: pd.Dataframe (n,m+1) containing m phenotypes and a weight column;
          wcol: str column of the weight in the df argument.
        Usage:
          ```
          # Define df, then get the weighted correlation:
          ```
          wcorr = WeightedCorr(xyw=df[["xcol", "ycol", "wcol"]])(method='pearson')
          ```
        """
        if (df is None) and (wcol is None):
            if np.all([i is None for i in [xyw, x, y, w]]):
                raise ValueError("No data supplied")

            if not (
                (isinstance(xyw, pd.DataFrame))
                != (np.all([isinstance(i, pd.Series) for i in [x, y, w]]))
            ):
                raise TypeError(
                    "xyw should be a pd.DataFrame, or x, y, w should be pd.Series"
                )

            xyw = pd.concat([x, y, w], axis=1).dropna() if xyw is None else xyw.dropna()
            self.x, self.y, self.w = (
                pd.to_numeric(xyw[i], errors="coerce").values for i in xyw.columns
            )
            self.df = None

        elif (wcol is not None) and (df is not None):
            if (not isinstance(df, pd.DataFrame)) or (not isinstance(wcol, str)):
                raise ValueError(
                    "df should be a pd.DataFrame and wcol should be a string"
                )

            if wcol not in df.columns:
                raise KeyError("wcol not found in column names of df")

            self.df = df.loc[:, [x for x in df.columns if x != wcol]]
            self.w = pd.to_numeric(df.loc[:, wcol], errors="coerce")

        else:
            raise ValueError(
                "Incorrect arguments specified, please specify xyw, or (x, y, w) or (df, wcol)"
            )

    def _wcov(self, x, y, ms):
        return np.sum(self.w * (x - ms[0]) * (y - ms[1]))

    def _pearson(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        mx, my = (np.sum(i * self.w) / np.sum(self.w) for i in [x, y])
        return self._wcov(x, y, [mx, my]) / np.sqrt(
            self._wcov(x, x, [mx, mx]) * self._wcov(y, y, [my, my])
        )

    def _wrank(self, x):
        (unique, arr_inv, counts) = np.unique(
            rankdata(x), return_counts=True, return_inverse=True
        )
        a = np.bincount(arr_inv, self.w)
        return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]

    def _spearman(self, x=None, y=None):
        x, y = (self.x, self.y) if ((x is None) and (y is None)) else (x, y)
        return self._pearson(self._wrank(x), self._wrank(y))

    def __call__(self, method: str = "pearson") -> Union[float, ]:
        """
        WeightedCorr call method.
        Args:
          method (str, "pearson"): Correlation method to be used:
                                   'pearson' for pearson r, 'spearman' for spearman rank-order correlation.
        Return:
          - The correlation value as float if xyw, or (x, y, w) were passed to __init__;
          - A m x m pandas.DataFrame holding the correlaton matrix if (df, wcol) were passed to __init__.
        """
        method = method.lower()
        if method not in CORR_METHODS:
            raise ValueError(f"`method` should be one of {CORR_METHODS}.")

        # define which of the defined methods to use:
        cor = {"pearson": self._pearson, "spearman": self._spearman}[method]
        if self.df is None:  # run the method over series
            return cor()
        else:
            # run the method over matrix
            out = pd.DataFrame(np.nan,
                               index=self.df.columns,
                               columns=self.df.columns)
            for i, x in enumerate(self.df.columns):
                for j, y in enumerate(self.df.columns):
                    if i >= j:
                        out.loc[x, y] = cor(
                            x=pd.to_numeric(self.df[x], errors="coerce"),
                            y=pd.to_numeric(self.df[y], errors="coerce"),
                        )
                        out.loc[y, x] = out.loc[x, y]
            return out
