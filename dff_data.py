"""Preprocess Dominick's Finer Foods (DFF) scanner data into demand and price matrices.

Expects three raw CSV files in the current directory:
  wber.csv  – beer category
  wbjc.csv  – bottled juice category
  wsdr.csv  – soft drinks category

Each file contains weekly store-level UPC sales records.  The script filters to
a single store, selects the 20 most-frequently sold UPCs across all categories,
aggregates to weekly totals, and writes:
  demand.csv  – (weeks, 20) integer movement (units sold)
  prices.csv  – (weeks, 20) price per unit
"""

import pandas as pd


# Load raw scanner data

df_ber = pd.read_csv('wber.csv')
df_bjc = pd.read_csv('wbjc.csv')
df_sdr = pd.read_csv('wsdr.csv')


# Filter to single store, valid records, and positive sales

def preprocess(df, main_store=126):
    """Keep only records for `main_store` that passed QC, have unit quantity, and positive movement."""
    df = df[df['STORE'] == main_store]
    df = df[df['OK'] == 1]      # quality flag
    df = df[df['QTY'] == 1]     # unit-quantity records only
    df = df[df['MOVE'] > 0]     # positive movement (units sold)
    return df.reset_index(drop=True)


df_ber_pre = preprocess(df_ber)
df_bjc_pre = preprocess(df_bjc)
df_sdr_pre = preprocess(df_sdr)
df_pre = pd.concat([df_ber_pre, df_bjc_pre, df_sdr_pre], ignore_index=True)


# Select the 20 most-frequently sold UPCs across all categories

top_upcs = df_pre['UPC'].value_counts().head(20).index
df = df_pre[df_pre['UPC'].isin(top_upcs)]

# Verify that each (UPC, WEEK) pair has a single price — a prerequisite for
# the demand model which assumes one price per product per period.
assert df.groupby(['UPC', 'WEEK']).agg(
    {'PRICE': lambda x: x.max() - x.min()}
)['PRICE'].value_counts().index.max() == 0.0, \
    "Multiple prices found for a (UPC, WEEK) pair."

# Aggregate: sum movement and take the single price within each (UPC, WEEK)
df = df.groupby(['UPC', 'WEEK']).agg({'MOVE': 'sum', 'PRICE': 'first'}).reset_index()


# Pivot to (weeks × products) matrices and save

# dropna(axis=0) removes weeks where any product has no sales (ensures complete panels)
demand = df.pivot(index='WEEK', columns='UPC', values=['MOVE']).dropna(axis=0)
prices = df.pivot(index='WEEK', columns='UPC', values=['PRICE']).dropna(axis=0)

demand.to_csv('demand.csv', index=False, header=False)
prices.to_csv('prices.csv', index=False, header=False)
