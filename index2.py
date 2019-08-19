import os
import sys
import tempfile

import six

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import summary_v2 as hp
from tensorboard.summary.writer.event_file_writer import EventFileWriter

ninf = float("-inf")
inf = float("inf")

#HP_X = hp.HParam('x', hp.RealInterval(ninf, inf))
#HP_Y = hp.HParam('y', hp.RealInterval(ninf, inf))
#M_LOSS = hp.Metric('loss')
#M_ACC = hp.Metric('acc')

#HPARAM_CONFIG = hp.hparams_config_pb(
#    hparams=[HP_X, HP_Y],
#    metrics=[M_LOSS, M_ACC]
#)

RUNS = [
    ("aaaa", {"x": 1.0, "y": 0, "z": "cat"}, {"loss": -1.0, "acc": 0.2}),
    ("bbbb", {"x": 1.1, "y": 1, "z": "dog"}, {"loss": -1.1, "acc": 0.3}),
    ("cccc", {"x": 1.2, "y": 2, "z": "bird"}, {"loss": -1.2, "acc": 0.4}),
    ("dddd", {"w": "cow", "x": 1.3}, {"loss": -1.2, "mAP": 0.2}),
]

class SummaryWriter(object):

    def __init__(self, logdir):
        self._writer = EventFileWriter(logdir)

    def add_summary(self, summary, step=None):
        event = event_pb2.Event(summary=summary, step=step)
        self._writer.add_event(event)

    def close(self):
        self._writer.close()

def main():
    logdir = tempfile.mkdtemp(prefix="guild-summaries-")
    log_experiment(RUNS, logdir)
    log_runs(RUNS, logdir)
    print("Wrote summaries to %s" % logdir)

def log_experiment(runs, logdir):
    hparams = all_hparams(runs)
    metrics = all_metrics(runs)
    writer = SummaryWriter(logdir)
    writer.add_summary(Experiment(hparams, metrics))
    writer.close()

def all_hparams(runs):
    all = {}
    for _id, hparams, _metrics in runs:
        for name, val in hparams.items():
            all.setdefault(name, set()).add(val)
    return all

def all_metrics(runs):
    all = set()
    for _id, _hparams, metrics in runs:
        all.update(metrics)
    return list(all)

def log_runs(runs, logdir):
    for id, hparams, metrics in runs:
        log_run(id, hparams, metrics, logdir)

def log_run(run_id, hparams, metrics, logdir):
    writer = SummaryWriter(os.path.join(logdir, run_id))
    writer.add_summary(Session(run_id, hparams))
    for tag, val in metrics.items():
        writer.add_summary(Scalar(tag, val))
    writer.close()

def Experiment(hparams, metrics):
    return hp.hparams_config_pb(
        hparams=[HParam(name, vals) for name, vals in hparams.items()],
        metrics=[Metric(name) for name in metrics]
    )

def HParam(name, vals):
    if all_numbers(vals):
        return hp.HParam(name, hp.RealInterval(
            float(min(vals)), float(max(vals))))
    else:
        return hp.HParam(name, hp.Discrete(vals))

def all_numbers(vals):
    return all((isinstance(val, (int, float)) for val in vals))

def Metric(tag):
    return hp.Metric(tag)

def Session(name, hparams):
    try:
        return hp.hparams_pb(hparams, trial_id=name)
    except TypeError:
        return LegacySession(name, hparams)

def LegacySession(group_name, hparams):
    import time
    hparams = hp._normalize_hparams(hparams)
    ssi = hp.plugin_data_pb2.SessionStartInfo(
        group_name=group_name,
        start_time_secs=time.time(),
    )
    for name in sorted(hparams):
        val = hparams[name]
        if isinstance(val, bool):
            ssi.hparams[name].bool_value = val
        elif isinstance(val, (float, int)):
            ssi.hparams[name].number_value = val
        elif isinstance(val, six.string_types):
            ssi.hparams[name].string_value = val
        else:
            assert False, (name, val)
    return hp._summary_pb(
        hp.metadata.SESSION_START_INFO_TAG,
        hp.plugin_data_pb2.HParamsPluginData(session_start_info=ssi),
    )

def Scalar(tag, val):
    return summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(
            tag=tag, simple_value=val)])


if __name__ == "__main__":
    main()
