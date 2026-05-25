import pandas as pd, matplotlib.pyplot as plt, pathlib, json, numpy as np
RES = pathlib.Path('.')
plt.rcParams['font.size']=12

def acc_vs_k():
    df = pd.read_csv('bow_grid.csv', header=None,
           names=['det','k','C','acc'])
    piv = df.pivot_table(index='k', columns='det', values='acc', aggfunc='max')
    piv.plot(marker='o')
    plt.ylabel('Train accuracy'); plt.xlabel('Vocabulary size (k)')
    plt.title('BoVW accuracy vs dictionary size')
    plt.tight_layout(); plt.savefig('figs/bow_acc_vs_k.pdf'); plt.savefig('figs/bow_acc_vs_k.png')

def main():
    pathlib.Path('figs').mkdir(exist_ok=True)
    acc_vs_k()
    print('All figures saved in figs/.')
if __name__=='__main__':
    main()
