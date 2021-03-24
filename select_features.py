import pandas as pd

def select_features(data, ordered_covariates_or_features, feature_set_indices):
	if isinstance(data, str):	# if the input named 'data' is a string (is a directory address)
		data = pd.read_csv(data)
	
	type_flag = 1	# flag for detecting feature type (0) or covariate type (1)
	output_data = pd.DataFrame()	# final dataframe to be returned

	for item in ordered_covariates_or_features:
		if ' t-' in item:	# there is ' t-' at the end of name
			type_flag = 0
			break
	
	if type_flag == 0:	# ordered_features
		features_to_be_selected  = []
		for index in feature_set_indices:
			features_to_be_selected.append(ordered_covariates_or_features[index])

		output_data = data[features_to_be_selected]

	elif type_flag == 1:	# ordered_covariates
		covariates_to_be_selected  = []
		for index in feature_set_indices:
			covariates_to_be_selected.append(ordered_covariates_or_features[index])

		for covariate_name in covariates_to_be_selected:
			# makes a dataframe (tmp_df) that contains columns containing substring <covariate_name>
			tmp_df = data.filter(regex=covariate_name)
			output_data_col_names = output_data.columns.tolist()
			output_data = pd.concat([output_data, tmp_df], axis=1, ignore_index=True)	# concat two dataframes
			output_data.columns = output_data_col_names + tmp_df.columns.tolist()
			print(output_data.columns)

	return output_data