import pandas as pd

file_names = ['cifar10', 'cifar10_final']
# models = ['resnet18', 'wrn_34_10']

for f in file_names:
    df = pd.read_csv(f'{f}.csv')
    df.sort_values(by = 'log_name', inplace=True, ascending=True)
    # df['model'] = df['model'].str.replace('/', '_')
    # df.sort_index() # 되돌리기

    # df.to_csv(f'{file_name}.csv', index=False)
    df.to_csv(f'{f}.csv', index=False)
