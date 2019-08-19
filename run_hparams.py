from tensorboardX import SummaryWriter

import sys
import time
import random

hparam = {'lr': [0.1, 0.01, 0.001],
          'bsize': [1, 2, 4],
          'n_hidden': [100, 200]}

metrics = {'accuracy', 'loss'}

def train(lr, bsize, n_hidden):
    x = random.random()
    return x, x*5

logdir = sys.argv[1] if len(sys.argv) > 1 else None

with SummaryWriter(logdir) as w:
    for lr in hparam['lr']:
        for bsize in hparam['bsize']:
            for n_hidden in hparam['n_hidden']:
                accu, loss = train(lr, bsize, n_hidden)
                hparams = {
                    'lr': lr,
                    'bsize': bsize,
                    'n_hidden': n_hidden
                }
                metrics = {
                    'accuracy': accu,
                    'loss': loss
                }
                w.add_hparams(hparams, metrics)

print("Wrote events to ./runs")
