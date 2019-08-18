# HParam support for 0.7

## Notes on HParam functionality in TensorBoard

### Experiment summary

The experiment summary specifies the hparams and metrics.

When only an experiment summary is specified (i.e. no scalars, no
sessions, etc.), TB does not display the HParams tab as active. When
the HParams tab is viewed (e.g. via the Inactive list) it does show
the hparams and metrics.
