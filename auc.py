import pandas as pd

def auc_score(prob, test_y):
    df = pd.DataFrame({'prob': prob, 'label': test_y})

    df = df.sort_values(by='prob')
    df = df.reset_index(drop=True)

    pos_count = df['label'].value_counts()[1]
    neg_count = df['label'].value_counts()[0]

    pos_rank = 0

    for i in range(df.index[-1], -1, -1):
        if df.iloc[i]['label'] == 1:
            pos_rank += i + 1

    auc_score = pos_rank - pos_count * (pos_count + 1) / 2
    auc_score = auc_score / (neg_count * pos_count)

    return auc_score