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

from __future__ import absolute_import, division, print_function

import os

from core.common.log import LOGGER
from sedna.common.class_factory import ClassType, ClassFactory
from models import HuggingfaceLLM, APIBasedLLM, VllmLLM

os.environ['BACKEND_TYPE'] = 'TORCH'

__all__ = ["BaseModel"]

@ClassFactory.register(ClassType.GENERAL, alias="EdgeModel")
class EdgeModel:
    """Models being deployed on the Edge
    """
    def __init__(self, **kwargs):
        """Initialize the CloudModel.

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. Details can be found in the `VllmLLM`, `HuggingfaceLLM`, `APIBasedLLM` class.

            Special keys:
            - `backend`: str, default "huggingface". The serving framework to be used.
        """

        LOGGER.info(kwargs)
        self.kwargs = kwargs
        self.model_name = kwargs.get("model", None)
        self.backend = kwargs.get("backend", "huggingface")
        self._set_config()

    def _set_config(self):
        """Set the model path in our environment variables due to Sedna’s [check](https://github.com/kubeedge/sedna/blob/ac623ab32dc37caa04b9e8480dbe1a8c41c4a6c2/lib/sedna/core/base.py#L132).
        """
        #
        os.environ["model_path"] = self.model_name

    def load(self, **kwargs):
        """Set the model backend to be used. Will be called by Sedna's JointInference interface.

        Raises
        ------
        Exception
            When the backend is not supported.
        """
        if self.backend == "huggingface":
            self.model = HuggingfaceLLM(**self.kwargs)
        elif self.backend == "vllm":
            self.model = VllmLLM(**self.kwargs)
        elif self.backend == "api":
            self.model = APIBasedLLM(**self.kwargs)
        else:
            raise Exception(f"Backend {self.backend} is not supported. Please use 'huggingface', 'vllm', or `api` ")

    def predict(self, data,  **kwargs):
        """Inference the model with the given data.

        Parameters
        ----------
        data : dict
            The data to be used for inference. See format at BaseLLM's `inference()`.
        kwargs : dict
            To Align with Sedna's JointInference interface.

        Returns
        -------
        dict
            Formatted Response. See `model._format_response()` for more details.
        """

        answer = self.model.inference(data)

        return answer

    def cleanup(self):
        """Save the cache and cleanup the model.
        """

        self.model.save_cache()
        self.model.cleanup()
