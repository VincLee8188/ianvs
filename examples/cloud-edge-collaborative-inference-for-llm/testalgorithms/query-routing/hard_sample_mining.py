# Copyright 2024 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hard Example Mining Algorithms"""

import abc
import random
from transformers import pipeline
from sedna.common.class_factory import ClassFactory, ClassType
from core.common.log import LOGGER

__all__ = ('BERTFilter', 'EdgeOnlyFilter', 'CloudOnlyFilter',
           'RandomRouterFilter', 'OracleRouterFilter')

class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __init__(self, **kwargs):
        LOGGER.info(f"USING {self.__class__.__name__}")

    def __call__(self, infer_result=None):
        """
        predict function, judge the sample is hard or not.

        Parameters
        ----------
        infer_result : array_like
            prediction result

        Returns
        -------
        is_hard_sample : bool
            `True` means hard sample, `False` means not.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1


@ClassFactory.register(ClassType.HEM, alias="BERTRouter")
class BERTFilter(BaseFilter, abc.ABC):
    def __init__(self, **kwargs):
        """Initialize the BERTFilter.

        Parameters
        ----------
        kwargs: dict
            Possible kwargs are:
            - `model`: str, default "routellm/bert". The model to be used.
            - `task`: str, default "text-classification". The task to be used.
            - `max_length`: int, default 512. The maximum length of the input.
        """
        super().__init__(**kwargs)

        self.kwargs = kwargs
        LOGGER.info(kwargs)

        self.model = kwargs.get("model", "routellm/bert")
        self.task = kwargs.get("task", "text-classification")
        self.max_length = kwargs.get("max_length", 512)

        self.classifier = pipeline(self.task, model=self.model, device="cuda")

    def _text_classification_postprocess(self, result):
        """Postprocess the text classification result

        Parameters
        ----------
        result : list
            The result from the classifier. Example:
            ```
            [{"label": "LABEL_0", "score": 0.5},
            {"label": "LABEL_1", "score": 0.4},
            {"label": "LABEL_2", "score": 0.1}]

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.
        """

        res = {item["label"]:item["score"] for item in result}
        scaled_score = res["LABEL_0"] / (res["LABEL_0"] + res["LABEL_1"])

        thresold = self.kwargs.get("threshold", 0.5)
        label = "LABEL_0" if scaled_score >= thresold else "LABEL_1"
        return False if label == "LABEL_0" else True

    def _predict(self, data):
        """Predict the data label

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.

        Raises
        ------
        NotImplementedError
            If the task is not supported
        """

        if self.task == "text-classification":
            result = self.classifier(data, top_k=None)
            is_hard_sample = self._text_classification_postprocess(result)
        else:
            raise NotImplementedError

        return is_hard_sample

    def _preprocess(self, data):
        """Preprocess the data

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        str
            query string
        """
        query = data.get("query")
        if "query" in query:
            return query["query"][:self.max_length]
        else:
            return query[:self.max_length]


    def cleanup(self):
        """Release the classifier model
        """
        del self.classifier

    def __call__(self, data=None) -> bool:
        data = self._preprocess(data)
        return self._predict(data)

@ClassFactory.register(ClassType.HEM, alias="EdgeOnly")
class EdgeOnlyFilter(BaseFilter, abc.ABC):
    """Route all queries to edge.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data=None) -> bool:
        return False

@ClassFactory.register(ClassType.HEM, alias="CloudOnly")
class CloudOnlyFilter(BaseFilter, abc.ABC):
    """Route all queries to cloud.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data=None) -> bool:
        return True

@ClassFactory.register(ClassType.HEM, alias="RandomRouter")
class RandomRouterFilter(BaseFilter, abc.ABC):
    """Randomly route the queries to edge or cloud.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = kwargs.get("threshold", 0)

    def __call__(self, data=None) -> bool:
        return False if random.random() < self.threshold else True

@ClassFactory.register(ClassType.HEM, alias="OracleRouter")
class OracleRouterFilter(BaseFilter, abc.ABC):
    """The Opitmal Router, which routes the queries to edge or cloud based on the models' prediction.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edge_better = 0
        self.cloud_better = 0
        self.both_right = 0
        self.both_wrong = 0

        self.edge_model = kwargs.get("edgemodel")
        self.cloud_model = kwargs.get("cloudmodel")

    def __call__(self, data=None):
        """Route the query to edge or cloud based on the models' prediction.

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.
        """
        gold = data.get("gold", None)

        edge_result = self.edge_model.predict(data).get("prediction")
        cloud_result = self.cloud_model.inference(data).get("prediction")

        both_right = edge_result == gold and cloud_result == gold
        both_wrong = edge_result != gold and cloud_result != gold
        edge_better = edge_result == gold and cloud_result != gold
        cloud_better = edge_result != gold and cloud_result == gold

        if both_right:
            self.both_right +=1
        elif both_wrong:
            self.both_wrong += 1
        elif edge_better:
            self.edge_better += 1
        elif cloud_better:
            self.cloud_better += 1

        if cloud_better:
            # cloud is better than edge, hard sample
            return True
        else:
            # both correct + both wrong + edge_better, easy sample
            return False

    def cleanup(self):
        """Leverage the `cleanup()` interface to print the statistics.
        """
        message = [
            f"OracleRouter Statistics: \n",
            f"Both Wrong: {self.both_wrong},  ",
            f"Both Correct: {self.both_right},  ",
            f"Edge Better: {self.edge_better},  ",
            f"Cloud Better: {self.cloud_better}"
        ]
        LOGGER.info("".join(message))
