def superiority_oos(TE_O_baseline, TE_O_model):
    return ((TE_O_baseline - TE_O_model) / TE_O_baseline) * 100