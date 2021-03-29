import sys

import pandas as pd

import configurations


def rank_covariates(data, ranking_method, forced_covariates):
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("The data input format is not valid.")
    data.drop(configurations.NOT_NUMERICAL_COLUMNS_NAMES, axis=1, inplace=True)

    corr = data.corr().abs()
    features_names = corr.index.drop([configurations.TARGET_COLUMN_NAME])
    overall_rank_df = pd.DataFrame(index=corr.index, columns=['mRMR_rank'])
    for ind in corr.index:
        overall_rank_df.loc[ind, 'mRMR_rank'] = corr.loc[ind, configurations.TARGET_COLUMN_NAME]\
                                                - corr.loc[ind, features_names].mean()
    overall_rank_df.sort_values(by='mRMR_rank', ascending=False, inplace=True)
    overall_rank = overall_rank_df.index.tolist()
    print(overall_rank_df)
    print(overall_rank)
    # final_rank = []
    # final_rank = overall_rank[0:2]
    # overall_rank = overall_rank[2:]
    # while len(overall_rank) > 0:
    #     temp = pd.DataFrame(index=overall_rank, columns=['mrmr_rank'])
    #     for i in overall_rank:
    #         temp.loc[i, 'mrmr_rank'] = cor.loc[i, 'target'] - cor.loc[i, final_rank[1:]].mean()
    #     temp = temp.sort_values(by='mrmr_rank', ascending=False)
    #     final_rank.append(temp.index[0])
    #     overall_rank.remove(temp.index[0])
    #
    # # next 6 lines arranges columns in order of correlations with target or by mRMR rank
    # if (feature_selection == 'mrmr'):
    #     final_rank.remove('target')
    #     ix = final_rank
    # else:
    #     ix = ranking_data.corr().abs().sort_values('target', ascending=False).index.drop(['target'])

    return


if __name__ == '__main__':
    rank_covariates('historical_data h=3.csv', 'mRMR', [])
