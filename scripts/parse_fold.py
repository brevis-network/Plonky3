import sys
import re
import pandas as pd

def main():
    transcript = sys.stdin.read()
    name_match = re.compile(r'(\S+)::<(\S+)>/\((.+)\)')
    time_match = re.compile(r'time:\s+\[\d+.\d+ \S+ (\d+.\d+ \S+) \d+.\d+ \S+\]')

    names = name_match.findall(transcript)
    times = time_match.findall(transcript)

    data = []

    for e, t in zip(names, times):
        method, field, params = e
        
        if re.match(r'(\d+), (\d+)', params):
            num_vars = int(params.split(',')[0])
            method = method + '/log(arity)=' + params.split(',')[1]
        else:
            num_vars = int(params)
            
        data.append([num_vars, field, method, t])

    df = pd.DataFrame(data, columns= ['num_vars', 'field', 'method', 'time'])
    num_vars = df['num_vars'].unique()
    field = df['field'].unique()
    method = df['method'].unique()

    df['method'] = pd.Categorical(df['method'], categories=method, ordered=True)

    df_idx = pd.MultiIndex.from_product([num_vars, field], names=['num_vars', 'field'])
    df_pivot = df.pivot(index = ('num_vars', 'field'), columns = 'method', values = 'time')
    df_pivot = df_pivot.reindex(df_idx).reset_index()

    df_pivot.to_csv(sys.stdout, index=False, sep='\t')

if __name__ == '__main__':
    main()