import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('golf_weather.csv')
    return df

def get_data_rainy():
    df = pd.read_csv('golf_weather.csv')
    df = df[df['outlook']=='rainy']
    return df

def get_data_overcast():
    df = pd.read_csv('golf_weather.csv')
    df = df[df['outlook']=='overcast']
    return df

def get_data_sunny():
    df = pd.read_csv('golf_weather.csv')
    df = df[df['outlook']=='sunny']
    return df


def get_entrop(ps):
    eps = np.finfo(np.float32).eps
    ent = -sum([(p+eps)*np.log(p+eps) for p in ps])
    return ent

def get_H_D(df, label):
    print('\n')

    ys = df[label].unique()

    count = dict()
    for y in ys:
        count[y] = 0

    for idx, row in df.iterrows():
        y = row[label]
        count[y] = count[y] + 1

    print('data', count)

    c = np.array(list(count.values()))
    c = c/c.sum()
    h = get_entrop(c)
    print('H(D)=', h)
    return h


def get_H_D_given_A(df, label, feature):
    print('\n')

    ys = df[label].unique()
    xs = df[feature].unique()

    count = dict()
    for x in xs:
        count[x] = dict()
        for y in ys:
            count[x][y] = 0

    for idx, row in df.iterrows():
        x = row[feature]
        y = row[label]
        count[x][y] = count[x][y] + 1

    print(feature, count)

    ws = []
    hs = []
    for x in xs:
        c = np.array(list(count[x].values()))
        w = c.sum()
        c = c/w
        h = get_entrop(c)
        print('value:', x, 'weight:', w, 'entropy:', h)

        ws.append(w)
        hs.append(h)
    ws = np.array(ws)
    ws = ws/ws.sum()
    hs = np.array(hs)
    h_mean = ws.dot(hs)
    print('H(D|A)=', h_mean)

    return h_mean



def main():
    df = get_data()
    get_H_D(df, 'play')
    get_H_D_given_A(df, 'play', 'outlook')
    get_H_D_given_A(df, 'play', 'humidity nomial')
    get_H_D_given_A(df, 'play', 'windy')

    df_sunny = get_data_sunny()
    get_H_D(df_sunny, 'play')
    get_H_D_given_A(df_sunny, 'play', 'humidity nomial')
    get_H_D_given_A(df_sunny, 'play', 'windy')

    df_overcast = get_data_overcast()
    get_H_D(df_overcast, 'play')

    df_rainy = get_data_rainy()
    get_H_D(df_sunny, 'play')
    get_H_D_given_A(df_rainy, 'play', 'humidity nomial')
    get_H_D_given_A(df_rainy, 'play', 'windy')


if __name__=='__main__':
    main()
