import pickle
import pandas as pd


test_df = pd.read_csv(f'data/input/test.csv')

with open(f'data/output/results/similar_ids.pickle', 'rb') as f:
    sim_ids_dict = pickle.load(f)

sub_df = test_df[['gid']].copy()
sub_df['cite_gid'] = sub_df['gid'].map(sim_ids_dict)

assert sub_df[sub_df['cite_gid'].isnull()].shape[0] == 0

sub_df['cite_gid'] = sub_df['cite_gid'].apply(lambda x: ' '.join(map(str,x[:20])))

print(sub_df.head())

sub_df.to_csv('data/output/results/test_submission.csv', index=False)