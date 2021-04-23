import sys

import pandas as pd

import configurations


def rank_features(data,
                  ranking_method: str,
                  forced_features: list):

    if isinstance(data, str):
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("The data input format is not valid.")

    data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)
    data.drop(forced_features, axis=1, inplace=True)

    cor = data.corr().abs()
    valid_feature = cor.index.drop([configurations.TARGET_COLUMN_NAME])
    overall_rank_df = pd.DataFrame(index=cor.index, columns=['mrmr_rank'])
    for i in cor.index:
        overall_rank_df.loc[i, 'mrmr_rank'] = \
            cor.loc[i, configurations.TARGET_COLUMN_NAME] - cor.loc[i, valid_feature].mean()
    overall_rank_df = overall_rank_df.sort_values(by='mrmr_rank', ascending=False)
    overall_rank = overall_rank_df.index.tolist()
    final_rank = overall_rank[0:2]
    overall_rank = overall_rank[2:]
    while len(overall_rank) > 0:
        temp = pd.DataFrame(index=overall_rank, columns=['mrmr_rank'])
        for i in overall_rank:
            temp.loc[i, 'mrmr_rank'] = cor.loc[i, configurations.TARGET_COLUMN_NAME] - cor.loc[i, final_rank[1:]].mean()
        temp = temp.sort_values(by='mrmr_rank', ascending=False)
        final_rank.append(temp.index[0])
        overall_rank.remove(temp.index[0])

    # next 6 lines arranges columns in order of correlations with target or by mRMR rank
    if ranking_method == 'mRMR':
        final_rank.remove(configurations.TARGET_COLUMN_NAME)
        ix = final_rank
    else:
        ix = data.corr().abs().sort_values(configurations.TARGET_COLUMN_NAME, ascending=False).index.drop(
            [configurations.TARGET_COLUMN_NAME])
    ranked_features = forced_features
    ranked_features.extend(ix)
    return ranked_features


if __name__ == '__main__':
    ranked_features = rank_features(data='historical_data h=2.csv',
                                      ranking_method='mRMR',
                                      forced_features=[])

    # print(len(ranked_features))
    # print(ranked_features)
