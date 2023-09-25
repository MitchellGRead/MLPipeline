from enum import Enum

from ray.data import Preprocessor

from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.metric_handler_interface import MetricHandlerInterface
from ml.api.model_interface import ModelInterface
from ml.data_handler.complex_physics import ComplexPhysicsDataHandler
from ml.data_handler.tagifai import TagifaiDataHandler
from ml.metric.tagifai import TagifaiMetricHandler
from ml.model.tagifai import TagifaiModel
from ml.preprocessor.tagifai import TagifaiPreprocessor

# Identifiable model ids for pipeline configuration
tagifai_model_id = "Tagifai_LLM_Model"
complex_gns_model_id = "Complex_Physics_GNS"


class ModelProject(Enum):
    LLM = "LLM"
    GNS = "GNS"


def _create_model_entry(
    model: ModelInterface,
    preprocessor: Preprocessor,
    data_handler: DataHandlerInterface,
    metric_handler: MetricHandlerInterface,
    model_project: ModelProject,
) -> dict[str, any]:
    return {
        "model": model,
        "preprocessor": preprocessor,
        "data_handler": data_handler,
        "metric_handler": metric_handler,
        "model_project": model_project,
    }


# Reference to models via their id containing model specific information and lazy loaded class instances
pipeline_models: dict[str, dict[str, any]] = {
    tagifai_model_id: _create_model_entry(
        TagifaiModel,
        TagifaiPreprocessor,
        TagifaiDataHandler,
        TagifaiMetricHandler,
        ModelProject.LLM,
    ),
    complex_gns_model_id: _create_model_entry(
        None, None, ComplexPhysicsDataHandler, None, ModelProject.GNS
    ),
}


def getModelFactory(for_model: str) -> ModelInterface:
    # return pipeline_models[for_model][0]
    return pipeline_models[for_model]["model"]


def getPreprocessorFactory(for_model: str) -> Preprocessor:
    return pipeline_models[for_model]["preprocessor"]


def getDataHandler(for_model: str) -> DataHandlerInterface:
    return pipeline_models[for_model]["data_handler"]


def getMetricHandler(for_model: str) -> MetricHandlerInterface:
    return pipeline_models[for_model]["metric_handler"]


def getModelProject(for_model: str) -> ModelProject:
    return pipeline_models[for_model]["model_project"]
