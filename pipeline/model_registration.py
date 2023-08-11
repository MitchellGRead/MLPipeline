from ray.data import Preprocessor

from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.model_interface import ModelInterface
from ml.data.tagifai_data_handler import TagifaiDataHandler
from ml.model.tagifai import TagifaiModel
from ml.preprocessor.tagifai import TagifaiPreprocessor

# Identifiable model ids for pipeline configuration
tagifai_model_id = "Tagifai_LLM_Model"

# Reference to models via their id lazily loaded
pipeline_models: dict[tuple[ModelInterface, Preprocessor, DataHandlerInterface]] = {
    tagifai_model_id: (TagifaiModel, TagifaiPreprocessor, TagifaiDataHandler),
}


def getModelFactory(for_model: str) -> ModelInterface:
    return pipeline_models[for_model][0]


def getPreprocessorFactory(for_model: str) -> Preprocessor:
    return pipeline_models[for_model][1]


def getDataHandler(for_model: str) -> DataHandlerInterface:
    return pipeline_models[for_model][2]
