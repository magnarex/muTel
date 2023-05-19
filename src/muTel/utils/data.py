from muTel.utils.meta import parent
from IPython.display import display
import pandas as pd
data_path = f'{parent}/data/'

def read_muon(run,nhits=None,sl=None):
    file_path = '/MuonData_{run}'.format(run=run)

    if nhits is not None:
        file_path += f'_{nhits}hits'
    if sl == 'all':
        file_path += '_allSL'
    elif sl is not None:
        file_path += f'_SL{sl}'

    return pd.read_csv(data_path+file_path+'.txt')


def display_df(self):
    with pd.option_context('display.max_rows',10):
        display(self.df)