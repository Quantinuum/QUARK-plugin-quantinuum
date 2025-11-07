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

from qnexus.models.references import CompilationResultRef
from quark.core import Core, Data, Result
from quark.interface_types import Other
import qnexus as qnx
import datetime

from ..interfaces.backend_result import BackendResult
from ..interfaces.benchmark_circuits_pytket import BenchmarkCircuitsPytket
from ..interfaces.nexus_compilation_result import NexusCompilationResult

logger = logging.getLogger()


@dataclass
class NexusCompilation(Core):
    device: str
    optimisation_level: int = 1
    project: str = "quark_benchmarking"

    @override
    def preprocess(self, input_data: Other[BenchmarkCircuitsPytket]) -> Result:
        project = qnx.projects.get_or_create(name=self.project)
        qnx.context.set_active_project(project)
        config = qnx.QuantinuumConfig(device_name=self.device)
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
        compile_job_ref = qnx.start_compile_job(
            programs=circuit_refs,
            backend_config=config,
            optimisation_level=self.optimisation_level,
            name=f"{benchmark_name}-compilation-job-{jobname_suffix}",
        )
        logger.info(
            f"Compiling circuits for benchmark {benchmark_name} on Nexus for device {self.device}"
        )
        qnx.jobs.wait_for(compile_job_ref)
        result_refs = qnx.jobs.results(compile_job_ref)
        compiled_circuits = list()
        for ref in result_refs:
            assert isinstance(ref, CompilationResultRef)
            compiled_circuits.append(ref.get_output())

        return Data(
            Other(
                NexusCompilationResult(
                    compiled_circuits, benchmark_name, self.device, self.project
                )
            )
        )

    @override
    def postprocess(self, input_data: Other[BackendResult]) -> Result:
        return Data(input_data)
