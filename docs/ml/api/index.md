## ML Pipeline API

The following docs outline the ML pipeline api. The following are required implementations that need to be established so they can plug into the pipeline and work as expected.

Currently the pipeline only supports Torch implementations so keep this in mind during development.

## Documentation

- [Data Handler](data_handler_interface.md): Data Handler Interface.
- [Metric Handler](metric_handler_interface.md): Metric Handler Interface.
- [Model](model_interface.md): Model Interface.
- [Preprocessor](preprocessor.md): Preprocessor Interface.

## Current Models

Currently developed model components

### Data

- [Utils](../data/utils.md): Data utils
- [Tagifai Data Handler](../data/tagifai.md): Tagifai Data Handler

### Metric

- [Tagifai Metric Handler](../metric/tagifai.md): Tagifai Metric Handler

### Model

- [Tagifai Model](../model/tagifai.md): Tagifai Model

### Preprocssor

- [Tagifai Preprocessor](../preprocessor/tagifai.md): Tagifai Preprocessor
