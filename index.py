from __future__ import print_function

import argparse
import logging
import sys
import tempfile

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

logging.basicConfig(format="%(message)s", level=logging.INFO)

log = logging.getLogger()

HPARAM_PLUGIN_NAME = "hparams"
HPARAM_DATA_VER = 0

EXPERIMENT_TAG = "_hparams_/experiment"
SESSION_START_INFO_TAG = '_hparams_/session_start_info'
SESSION_END_INFO_TAG = '_hparams_/session_end_info'

class SampleRun(object):

    def __init__(self, opdef):
        self.id = runlib.mkid()
        self.opref = opreflib.OpRef(
            "guildfile", opdef.guildfile.src, "",
            opdef.modeldef.name, opdef.name)
        self.short_id = self.id[:8]
        self.status = "pending"
        self._attrs = {
            "flags": {
                "noise": 0.1,
                "x": 1.0,
            }
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
    logdir = args.logdir or tempfile.mkdtemp(prefix="guild-summaries-")
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
    for name, _, desc in CMDS:
        print(name.ljust(10), desc)
    raise SystemExit()

def _default(gf, logdir):
    log.info("Running default scenario")
    opdef = _opdef(gf, "noisy")
    run = SampleRun(opdef)
    run.start()
    scalars = run_scalars(run)
    with SummaryWriter(logdir) as writer:
        add_experiment(writer, opdef.flags, ["loss"])
        add_session_start_info(writer, run)
        add_scalars(writer, scalars)
        add_session_end_info(writer, run)

def _opdef(gf, name):
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

def add_scalars(writer, scalars):
    log.info(" - Scalars for %i value(s)", len(scalars))
    for tag, value, step in scalars:
        writer.add_scalar(tag, value, step)

def add_experiment(writer, flagdefs, scalars):
    log.info(
        " - Experiment with %i flag(s) and %i metric(s)",
        len(flagdefs), len(scalars))
    _add_summary(writer, _ExperimentSummary(flagdefs, ["loss"]))

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

def _ExperimentSummary(flags, scalars):
    experiment = _Experiment(flags, scalars)
    return _HParamSummary(EXPERIMENT_TAG, _HParamExperimentData(experiment))

def _Experiment(flags, scalars):
    return Experiment(
        hparam_infos=[_HParamInfo(flag) for flag in flags],
        metric_infos=[_MetricInfo(name) for name in scalars])

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
    return HParamsPluginData(experiment=experiment, version=HPARAM_DATA_VER)

def _HParamSummary(tag, data):
    metadata = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=HPARAM_PLUGIN_NAME,
            content=data.SerializeToString()))
    return Summary(value=[Summary.Value(tag=tag, metadata=metadata)])

def _SessionStartInfoSummary(run):
    info = _SessionStartInfo(run)
    return _HParamSummary(
        SESSION_START_INFO_TAG, _HParamSessionStartInfoData(info))

def _SessionStartInfo(run):
    flags = run.get("flags") or {}
    started_secs = _safe_seconds(run.get("started"))
    session = SessionStartInfo(
        model_uri="not sure what this is",
        group_name=_session_group_name(run),
        start_time_secs=started_secs)
    for name, val in flags.items():
        _apply_session_hparam(val, name, session)
    return session

def _safe_seconds(timestamp):
    if timestamp is None:
        return None
    return timestamp / 1000000

def _session_group_name(run):
    operation = run_util.format_operation(run)
    label_part = _label_part(run.get("label"))
    return "%s %s%s" % (run.short_id, operation, label_part)

def _label_part(label):
    if not label:
        return ""
    return " " + label

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
    return HParamsPluginData(session_start_info=info, version=HPARAM_DATA_VER)

def _SessionEndInfoSummary(run):
    info = _SessionEndInfo(run)
    return _HParamSummary(SESSION_END_INFO_TAG, _HParamSessionEndInfoData(info))

def _SessionEndInfo(run):
    end_secs = _safe_seconds(run.get("stopped"))
    return SessionEndInfo(status=_Status(run), end_time_secs=end_secs)

def _Status(run):
    if run.status in ("terminated", "completed"):
        return Status.STATUS_SUCCESS
    elif run.status == "error":
        return Status.STATUS_FAILURE
    elif run.status == "running":
        return Status.STATUS_RUNNING
    else:
        return Status.STATUS_UKNOWN

def _HParamSessionEndInfoData(info):
    return HParamsPluginData(session_end_info=info, version=HPARAM_DATA_VER)

CMDS = [
    ("default", _default, "default scenario"),
]

if __name__ == "__main__":
    main()
