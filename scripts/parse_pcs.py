import sys
import re
import pandas as pd

def main():
    transcript = sys.stdin.read()
    name_match = re.compile(r'fri/(\S+)/test_shapes=(.+)/log_arity=(\d+)')
    time_match = re.compile(r'\S+:\s+\[\d+.\d+ \S+ (\d+.\d+ \S+) \d+.\d+ \S+\]')

    names = name_match.findall(transcript)
    times = time_match.findall(transcript)

    data = []

    for e, t in zip(names, times):
        method, shapes, log_arity = e
        data.append([log_arity, method, shapes, t])

    df = pd.DataFrame(data, columns= ['log(arity)', 'method', 'shapes', 'time'])
    shapes = df['shapes'].unique()
    log_arity = df['log(arity)'].unique()
    method = df['method'].unique()

    df['method'] = pd.Categorical(df['method'], categories=method, ordered=True)

    df_idx = pd.MultiIndex.from_product([shapes, log_arity], names=['shapes', 'log(arity)'])
    df_pivot = df.pivot(index = ['shapes', 'log(arity)'], columns = 'method', values = 'time')
    df_pivot = df_pivot.reindex(df_idx).reset_index()

    df_pivot.to_csv(sys.stdout, index=False, sep='\t')

if __name__ == '__main__':
    main()