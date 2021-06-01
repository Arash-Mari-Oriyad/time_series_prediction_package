import pandas as pd

mbp_data = pd.read_csv('main_historical_data h=1.csv')

print(mbp_data.shape)
print(mbp_data['spatial id'].nunique())

temp_spatial_ids = mbp_data['spatial id'].values.tolist()[:10]

changed_mbp_data = mbp_data[mbp_data['spatial id'].isin(temp_spatial_ids)].copy()

print(changed_mbp_data.shape)
print(changed_mbp_data['spatial id'].nunique())

changed_mbp_data.to_csv('historical_data h=1.csv', index=False)
