import rl
from strat import canon, do_one, chain

def subs(d):
    """ Full simultaneous exact substitution """
    # TODO: Find a better way to ensure that we don't accidentaly match
    enum = {v: 2942742429+i for i,v in enumerate(d.values())}
    step1 = {k: enum[v] for k,v in d.items()}
    step2 = {enum[v]: v for k,v in d.items()}
    subs1 = do_one(*map(rl.subs, *zip(*step1.items())))
    subs2 = do_one(*map(rl.subs, *zip(*step2.items())))

    return chain(canon(subs1), canon(subs2))
