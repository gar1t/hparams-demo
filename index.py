from __future__ import print_function

import argparse
import logging
import os
import random
import sys
import tempfile
import time

from tensorboardX import SummaryWriter

from tensorboardX.proto.api_pb2 import Experiment
from tensorboardX.proto.api_pb2 import HParamInfo
from tensorboardX.proto.api_pb2 import Interval
from tensorboardX.proto.api_pb2 import MetricInfo
from tensorboardX.proto.api_pb2 import MetricName
from tensorboardX.proto.api_pb2 import Status

from tensorboardX.proto.plugin_hparams_pb2 import HParamsPluginData
from tensorboardX.proto.plugin_hparams_pb2 import SessionEndInfo
from tensorboardX.proto.plugin_hparams_pb2 import SessionStartInfo

from tensorboardX.proto.summary_pb2 import Summary
from tensorboardX.proto.summary_pb2 import SummaryMetadata

from tensorboardX.proto.types_pb2 import DT_FLOAT

from tensorboardX.x2num import make_np

from guild import guildfile
from guild import opref as opreflib
from guild import run as runlib
from guild import run_util
from guild import util

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO)

log = logging.getLogger()

HPARAM_PLUGIN_NAME = "hparams"
HPARAM_DATA_VER = 0

EXPERIMENT_TAG = "_hparams_/experiment"
SESSION_START_INFO_TAG = '_hparams_/session_start_info'
SESSION_END_INFO_TAG = '_hparams_/session_end_info'

SAMPLE_FLAGS = {
    "noise": 0.1,
    "x": 1.0,
}

###################################################################
# Init
###################################################################

class SampleRun(object):

    def __init__(self, opdef, flags=None):
        self.id = runlib.mkid()
        self.opref = opreflib.OpRef(
            "guildfile", opdef.guildfile.src, "",
            opdef.modeldef.name, opdef.name)
        self.short_id = self.id[:8]
        self.status = "pending"
        self._attrs = {
            "flags": flags or SAMPLE_FLAGS
        }
        self.get = self._attrs.get
        self.guild_path = lambda _: "__not_uses__"

    def start(self):
        self._attrs["started"] = runlib.timestamp()
        self.status = "running"

    def stop(self, status="completed"):
        self._attrs["stopped"] = runlib.timestamp()
        self.status = status

def main():
    args = _init_args()
    handler = _cmd_handler(args)
    gf = guildfile.from_dir(".")
    logdir = _init_logdir(args)
    handler(gf, logdir)
    log.info("Wrote summaries to %s", logdir)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("logdir", nargs='?')
    return p.parse_args()

def _cmd_handler(args):
    if args.cmd == "help":
        _print_help_and_exit()
    for name, handler, _desc in CMDS:
        if args.cmd == name:
            return handler
    raise SystemExit(
        "{prog}: invalid cmd '{cmd}'\n"
        "Try 'python {prog} help' for a list of commands."
        .format(prog=sys.argv[0], cmd=args.cmd))

def _print_help_and_exit():
    max_name = max([len(cmd[0]) for cmd in CMDS])
    for name, _, desc in CMDS:
        print(name.ljust(max_name + 1), desc)
    raise SystemExit()

def _init_logdir(args):
    return args.logdir or tempfile.mkdtemp(prefix="guild-summaries-")

###################################################################
# Commands
###################################################################

def _default(gf, logdir):
    log.info("Running default scenario")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)
    run.start()
    scalars = run_scalars(run)
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        add_session_start_info(writer, run)
        add_scalars(writer, scalars)
        add_session_end_info(writer, run)

def _status_change_session(gf, logdir):
    log.info("Running status change 'by session' scenario (fails)")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)

    def session():
        add_session_start_info(writer, run)
        add_session_end_info(writer, run)

    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        session()
        log.info(" - Starting run")
        run.start()
        session()
        log.info(" - Stopping run")
        run.stop()
        session()

def _status_change_experiment(gf, logdir):
    log.info("Running status change 'by experiment' scenario (fails)")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)

    def experiment():
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        add_session_start_info(writer, run)
        add_session_end_info(writer, run)

    with SummaryWriter(logdir) as writer:
        experiment()
        log.info(" - Starting run")
        run.start()
        experiment()
        log.info(" - Stopping run")
        run.stop()
        experiment()

def _status_change_summary(gf, logdir):
    log.info("Running status change 'by summary' scenario (fails)")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)

    def summary():
        time.sleep(1) # ensure we get a unique timestamp in logdir
        with SummaryWriter(logdir) as writer:
            add_experiment(writer, opdef.flags, scalar_tags(scalars))
            add_session_start_info(writer, run)
            add_session_end_info(writer, run)

    summary()
    log.info(" - Starting run")
    run.start()
    summary()
    log.info(" - Stopping run")
    run.stop()
    summary()

def _status_change_replace(gf, logdir):
    log.info("Running status change 'by replace' scenario (fails)")
    log.info("Check status in %s", logdir)
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)

    def summary():
        log.info(" - Clearing log dir of event files")
        clear_dir(logdir)
        with SummaryWriter(logdir) as writer:
            add_experiment(writer, opdef.flags, scalar_tags(scalars))
            add_session_start_info(writer, run)
            add_session_end_info(writer, run)

    summary()
    pause("Run status should be UKNOWN - press Enter to replace")
    log.info(" - Starting run")
    run.start()
    summary()
    pause("Run status should be RUNNING - press Enter to replace")
    log.info(" - Stopping run")
    run.stop()
    summary()

def _no_session(gf, logdir):
    log.info("Running no-session scenario")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)
    scalars = run_scalars(run)
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        add_scalars(writer, scalars)

def _no_experiment(gf, logdir):
    log.info("Running no-experiment scenario")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)
    scalars = run_scalars(run)
    with SummaryWriter(logdir) as writer:
        add_session_start_info(writer, run)
        #add_session_end_info(writer, run)

def _check_status(gf, logdir):
    log.info("Running check-status scenario")
    log.info("Check status in %s", logdir)
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef)
    scalars = run_scalars(run)
    run.start()  # Does nothing as initial status not
                 # logged, but here to show life cycle
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        add_session_start_info(writer, run)
        writer.flush()
        pause("Run status should be UNKNOWN - press Enter to set")
        run.stop()
        add_session_end_info(writer, run)

def _latent_metrics(gf, logdir):
    log.info("Running check-status scenario (fails)")
    opdef = OpDef(gf, "noisy")
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, [])
        add_experiment(writer, {}, ["loss"])

def _runs(gf, logdir):
    log.info("Running runs scenario")
    opdef = OpDef(gf, "noisy")
    runs = [SampleRun(opdef, random_noisy_flags()) for _ in range(10)]
    for run in runs:
        _add_run_default(run, opdef, logdir)

def _add_run_default(run, opdef, logdir, scalars=None):
    log.info(" - Adding run %s", run.short_id)
    run_logdir = os.path.join(logdir, run_label(run))
    util.ensure_dir(run_logdir)
    if scalars is None:
        scalars = perturb_scalars(run_scalars(run))
    with SummaryWriter(run_logdir) as writer:
        add_scalars(writer, scalars)
        add_experiment(writer, opdef.flags, scalar_tags(scalars))
        add_session_start_info(writer, run)
        add_session_end_info(writer, run)
    return run_logdir

def _add_run(gf, logdir):
    log.info("Running add-run scenario")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef, random_noisy_flags())
    _add_run_default(run, opdef, logdir)

def _latent_metrics_2(gf, logdir):
    log.info("Running latent metrics v2 scenario (fails)")
    log.info("Check status in %s", logdir)
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef, random_noisy_flags())
    run_logdir = _add_run_default(run, opdef, logdir, [])
    pause(
        "Added run %s without metrics - press Enter to update"
        % run.short_id)
    clear_dir(run_logdir)
    _add_run_default(run, opdef, logdir)

def _latent_metrics_3(gf, logdir):
    log.info("Running latent metrics v3 scenario (fails)")
    opdef = OpDef(gf, "noisy")
    run = SampleRun(opdef, random_noisy_flags())
    scalars = run_scalars(run)
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, [], "1")
        add_experiment(writer, opdef.flags, scalar_tags(scalars), "2")

CMDS = [
    ("default",                  _default, "default scenario"),
    ("status-change-session",    _status_change_session,
     "update run status by session (fails)"),
    ("status-change-experiment", _status_change_experiment,
     "update run status by experiment (fails)"),
    ("status-change-summary",    _status_change_summary,
     "update run status by summary (fails)"),
    ("status-change-replace",    _status_change_replace,
     "update run status by replace (fails)"),
    ("no-session",               _no_session,
     "log only an experiment and scalars"),
    ("no-experiment",            _no_experiment,
     "log only session"),
    ("check-status",             _check_status,
     "pause to check status before setting"),
    ("latent-metrics",           _latent_metrics,
     "add metrics after adding experiment (fails)"),
    ("runs",                     _runs,
     "generate multiple runs"),
    ("add-run",                  _add_run,
     "add run to logdir"),
    ("latent-metrics-2",         _latent_metrics_2,
     "add matrics after adding experiment v2 (fails)"),
    ("latent-metrics-3",         _latent_metrics_3,
     "add metrics by adding multiple experiments"),
]

###################################################################
# Scenario support
###################################################################

def OpDef(gf, name):
    opdef = gf.default_model.get_operation("noisy")
    if not opdef:
        raise SystemExit("missing op '%s' in guild.yml" % name)
    return opdef

def run_scalars(_run):
    return [
        ("loss", 1.0, 1),
        ("loss", -0.4, 2),
        ("loss", -0.6, 3),
        ("loss", -0.7, 4),
    ]

def random_noisy_flags():
    return {
        "noise": round(0.1 + (random.uniform(-0.1, 0.2)), 4),
        "x": round(random.uniform(-3.0, 3.0), 4),
    }

def perturb_scalars(scalars):
    return [(tag, perturb_val(val), step) for tag, val, step in scalars]

def perturb_val(x):
    return x + random.uniform(-0.5, 0.5)

def add_scalars(writer, scalars):
    log.info(" - Scalars for %i value(s)", len(scalars))
    for tag, value, step in scalars:
        writer.add_scalar(tag, value, step)

def add_experiment(writer, flagdefs, scalar_tags, name=None):
    log.info(
        " - Experiment with %i flag(s) and %i metric(s)",
        len(flagdefs), len(scalar_tags))
    _add_summary(writer, _ExperimentSummary(flagdefs, scalar_tags, name))

def add_session_start_info(writer, run):
    log.info(
        " - Session start info for run '%s'",
        run_util.format_operation(run))
    _add_summary(writer, _SessionStartInfoSummary(run))

def add_session_end_info(writer, run):
    log.info(
        " - Session end info for run '%s' (status=%s)",
        run_util.format_operation(run), run.status)
    _add_summary(writer, _SessionEndInfoSummary(run))

def _add_summary(writer, s):
    writer._get_file_writer().add_summary(s)

def pause(prompt):
    print(prompt, end=" ")
    sys.stdin.readline()

def scalar_tags(scalars):
    tags = set()
    for tag, _val, _step in scalars:
        tags.add(tag)
    return list(tags)

def clear_dir(dir):
    for name in os.listdir(dir):
        os.remove(os.path.join(dir, name))

###################################################################
# HParam proto support
###################################################################

def _ExperimentSummary(flags, scalar_tags, name=None):
    experiment = _Experiment(flags, scalar_tags, name)
    return _HParamSummary(
        EXPERIMENT_TAG, _HParamExperimentData(experiment))

def _Experiment(flags, scalar_tags, name=None):
    return Experiment(
        name=name,
        hparam_infos=[_HParamInfo(flag) for flag in flags],
        metric_infos=[_MetricInfo(tag) for tag in scalar_tags])

def _HParamInfo(flag):
    return HParamInfo(
        name=flag.name,
        description=flag.description,
        type=_HParamType(flag),
        domain_interval=_HParamInterval(flag))

def _HParamType(flag):
    if not flag.type:
        return None
    if flag.type == "float":
        return DT_FLOAT
    elif flag.type == "int":
        return DT_INT32
    else:
        return DT_STRING

def _HParamInterval(flag):
    if flag.min is None and flag.max is None:
        return None
    return Interval(min_value=flag.min, max_value=flag.max)

def _MetricInfo(scalar_name):
    return MetricInfo(name=MetricName(tag=scalar_name))

def _HParamExperimentData(experiment):
    return HParamsPluginData(
        experiment=experiment, version=HPARAM_DATA_VER)

def _HParamSummary(tag, data):
    metadata = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=HPARAM_PLUGIN_NAME,
            content=data.SerializeToString()))
    return Summary(
        value=[Summary.Value(tag=tag, metadata=metadata)])

def _SessionStartInfoSummary(run):
    info = _SessionStartInfo(run)
    return _HParamSummary(
        SESSION_START_INFO_TAG,
        _HParamSessionStartInfoData(info))

def _SessionStartInfo(run):
    flags = run.get("flags") or {}
    started_secs = _safe_seconds(run.get("started"))
    session = SessionStartInfo(
        model_uri="not sure what this is",
        group_name=run_label(run),
        start_time_secs=started_secs)
    for name, val in flags.items():
        _apply_session_hparam(val, name, session)
    return session

def _safe_seconds(timestamp):
    if timestamp is None:
        return None
    return timestamp / 1000000

def run_label(run):
    operation = run_util.format_operation(run)
    return "%s %s" % (run.short_id, operation)

def _apply_session_hparam(val, name, session):
    if isinstance(val, (int, float)):
        session.hparams[name].number_value = make_np(val)[0]
    elif isinstance(val, six.string_types):
        session.hparams[name].string_value = val
    elif isinstance(val, bool):
        session.hparams[name].bool_value = val
    else:
        assert False, (name, val)

def _HParamSessionStartInfoData(info):
    return HParamsPluginData(
        session_start_info=info,
        version=HPARAM_DATA_VER)

def _SessionEndInfoSummary(run):
    info = _SessionEndInfo(run)
    return _HParamSummary(
        SESSION_END_INFO_TAG,
        _HParamSessionEndInfoData(info))

def _SessionEndInfo(run):
    end_secs = _safe_seconds(run.get("stopped"))
    return SessionEndInfo(
        status=_Status(run),
        end_time_secs=end_secs)

def _Status(run):
    if run.status in ("terminated", "completed"):
        return Status.STATUS_SUCCESS
    elif run.status == "error":
        return Status.STATUS_FAILURE
    elif run.status == "running":
        return Status.STATUS_RUNNING
    else:
        return Status.STATUS_UNKNOWN

def _HParamSessionEndInfoData(info):
    return HParamsPluginData(
        session_end_info=info,
        version=HPARAM_DATA_VER)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("")
