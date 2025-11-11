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
from typing import override
import logging

from quark.core import Core, Data, Result
from quark.interface_types import Other
import qnexus as qnx
import datetime

from ..interfaces.backend_result import BackendResult
from ..interfaces.benchmark_circuits_pytket import BenchmarkCircuitsPytket
from ..interfaces.nexus_upload_result import NexusUploadResult

logger = logging.getLogger()


@dataclass
class NexusUpload(Core):
    project: str = "quark_benchmarking"

    @override
    def preprocess(self, input_data: Other[BenchmarkCircuitsPytket]) -> Result:
        project = qnx.projects.get_or_create(name=self.project)
        qnx.context.set_active_project(project)
        jobname_suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
        backend_input = input_data.data
        benchmark_name = backend_input.benchmark_name
        circuits = backend_input.circuits
        circuit_refs = [
            qnx.circuits.upload(
                circuit=circuit, name=f"{benchmark_name}-{jobname_suffix}-{i}"
            )
            for i, circuit in enumerate(circuits)
        ]

        return Data(
            Other(
                NexusUploadResult(
                    circuit_refs,
                    circuits,
                    benchmark_name,
                    self.project,
                )
            )
        )

    @override
    def postprocess(self, input_data: Other[BackendResult]) -> Result:
        return Data(input_data)
