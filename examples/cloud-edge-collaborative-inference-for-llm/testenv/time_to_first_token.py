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

from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Time to First Token")
def time_to_first_token(_, y_pred):
    """Calculate the Time to First Token of the system.

    Parameters
    ----------
    _ :
        Ignored
    y_pred : list
        List of predictions from the JointInference paradigm

    Returns
    -------
    float
        Average Time to First Token (s) of the system
    """

    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    average_ttft = sum([pred.result.time_to_first_token for pred in infer_res]) / len(infer_res)

    return round(average_ttft, 3)