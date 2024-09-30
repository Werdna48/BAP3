#%% Environment imports
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom, beta
import preliz as pz
from bap_config import setup_bap3_config
setup_bap3_config()

# %% function for probabilities of sets
def P(S, A):
    '''
    Takes in the Sample space of events S 
    Take in the events of A

    If possible, computes the probability of A in S, otherwise 0 (as its impossible)
    '''
    if set(A).issubset(set(S)):
        return len(A)/len(S)
    else:
        return 0
    

# For example

S = ['a', 'b', 'c', 'd', 'e']
A_1 = ['a', 'b']
A_2 = ['f']


print(P(S, A_1)) # 40% probability

print(P(S, A_2)) # 0% probability f is not in A_2
# %%
pz.BetaBinomial(alpha=10, beta=10, n=6).plot_interactive()

# %%
