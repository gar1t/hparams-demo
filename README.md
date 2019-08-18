# HParam support for 0.7

## Notes on HParam functionality in TensorBoard

### Experiment summary

The experiment summary specifies the hparams and metrics.

When only an experiment summary is specified (i.e. no scalars, no
sessions, etc.), TB does not display the HParams tab as active. When
the HParams tab is viewed (e.g. via the Inactive list) it does show
the hparams and metrics.

As soon as a scalar value for a metric is logged, the HParams tab
appears. The hparams and and metric columns appear in the Table
View. However, as there are no sessions, there's no data displayed
anywhere under the HParams tab.

See `no-session` scenario for `index.py` demo.

### Session status

If a session is not added, a run will be marked as 'unknown'
status. Once a session end info summary is logged, the run status is
forever locked to the specified status.

There does not appear to be a way to denote a 'running' status and
then update later to something else.

See the various `change-status-*` scenarios that attempt to change the
run status.

The only apparent way to support this is to replace the summary logs
and restart the TensorBoard backend.

See `check-status` scenario for the one success path (though does not
support a 'running' status).

### Ending sessions

It's not clear that ending a session provides any value beyond setting
the session status. The end time doesn't appear to be used.

### Latent metrics

We don't know the list of metrics ahead of time and so any runs that
are in-flight can add more scalars. It's common, e.g. to print a final
metric at the end of a run.

If we wait until a run is finished before showing metrics, runs in
flight won't show up in the HParams tab. This is super lame.

I think the best way to handle this is to wait until there's at least
one top-level scalar available for a run - and then add the
experiment.

We could even wait another cycle before creating the experiment (e.g
using the pending scheme ala images). Though I think KISS for this
first pass.

We can certainly miss metrics here! The only sure fire way to deal
with this is to either require the list of metrics ahead of time or to
get TB to update its damn list of metrics.

Until then we can start with this approach.
