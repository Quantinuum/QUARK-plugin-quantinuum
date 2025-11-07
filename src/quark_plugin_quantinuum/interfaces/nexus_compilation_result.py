# Copyright Quantinuum
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
from dataclasses import dataclass
from qnexus.models.references import CircuitRef


@dataclass
class NexusCompilationResult:
    """
    Input required for simulation benchmarks
    submitted to a quantinuum nexus backend.
    """

    compiled_circuits: list[CircuitRef]
    benchmark_name: str
    device_name: str
    project_name: str
