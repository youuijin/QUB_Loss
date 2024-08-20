import pandas as pd

file_names = ['resnet18']
dir_names = ['csvs/env1', 'csvs/env2', 'csvs/env3/cifar10', 'csvs/env4/cifar10', 'csvs/env4/cifar100']
# models = ['resnet18', 'wrn_34_10']

for d in dir_names:
    for f in file_names:
        df = pd.read_csv(f'{d}/{f}.csv')
        df.sort_values(by = ['method', 'log_name'], inplace=True, ascending=[True, True])
        # df['model'] = df['model'].str.replace('/', '_')
        # df.sort_index() # 되돌리기

        # df.to_csv(f'{file_name}.csv', index=False)
        df.to_csv(f'{d}/{f}.csv', index=False)

df = pd.read_csv(f'test.csv')
# sorted_df = df.sort_values(by=['column1', 'column2'], inplace)
df.sort_values(by = ['test_eps', 'env', 'log_name'], ascending=[True, True, True], inplace=True)
# df['model'] = df['model'].str.replace('/', '_')
# df.sort_index() # 되돌리기

df.to_csv(f'test.csv', index=False)

df = pd.read_csv(f'test_black.csv')
# sorted_df = df.sort_values(by=['column1', 'column2'], inplace)
df.sort_values(by = ['test_eps', 'env', 'log_name'], ascending=[True, True, True], inplace=True)
# df['model'] = df['model'].str.replace('/', '_')
# df.sort_index() # 되돌리기

df.to_csv(f'test_black.csv', index=False)