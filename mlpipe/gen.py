import pandas as pd
def disc(df,var,bin):
	df[str(var)+'bin'] = pd.cut(df.var, bins=15, labels=False)
	return df[str(var)+'bin']