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
from dataclasses import dataclass, field
from typing import override, Any
import logging
import time

from pytket.circuit import BasisOrder
from qnexus.models.references import (
    ExecutionResultRef,
    CompilationResultRef,
    CircuitRef,
    ProjectRef,
    HUGRRef,
    QIRRef,
)
from quark.core import Core, Data, Result
from quark.interface_types import Other
import qnexus as qnx
import datetime

from .helpers import counter_key_to_string_key
from ..interfaces.nexus_upload_result import NexusUploadResult
from ..interfaces.backend_result import BackendResult
from ..interfaces.nexus_compilation_result import NexusCompilationResult

logger = logging.getLogger()


@dataclass
class QuantinuumNexus(Core):
    device: str
    compile_optimization_level: int = 1
    n_shots: int = 100
    project_name_override: str | None = None
    metrics: dict[str, Any] = field(init=False, default_factory=dict)
    _results: BackendResult | None = field(init=False, default=None)

    def __post_init__(self):
        self._nexus_device_config = qnx.QuantinuumConfig(device_name=self.device)

    @override
    def preprocess(self, input_data: Other[NexusUploadResult]) -> Result:
        nexus_upload_result = input_data.data
        nexus_project_name = (
            nexus_upload_result.nexus_project
            if self.project_name_override is None
            else self.project_name_override
        )
        project = qnx.projects.get_or_create(name=nexus_project_name)
        jobname_suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
        compilation_result = self.compile(nexus_upload_result, project, jobname_suffix)
        counts_per_circuit = self.run_compiled_circuits(
            compilation_result,
            nexus_upload_result.shots_per_circuit,
            project,
            jobname_suffix,
        )
        self._results = BackendResult(counts=counts_per_circuit)
        return Data(None)

    @override
    def postprocess(self, input_data: Data) -> Result:
        return Data(Other(self._results))

    def compile(
        self,
        nexus_upload_result: NexusUploadResult,
        project: ProjectRef,
        jobname_suffix: str,
    ) -> NexusCompilationResult:
        logger.info(
            f"Compiling circuits for benchmark {nexus_upload_result.benchmark_name} on Nexus {self.device}"
        )
        start = time.perf_counter()
        compile_job_ref = qnx.start_compile_job(
            programs=nexus_upload_result.circuit_refs,
            backend_config=self._nexus_device_config,
            optimisation_level=self.compile_optimization_level,
            project=project,
            name=f"{nexus_upload_result.benchmark_name}-compilation-job-{jobname_suffix}",
        )
        qnx.jobs.wait_for(compile_job_ref)
        end = time.perf_counter()
        self.metrics.update(
            {
                "compile_job": {
                    "id": str(compile_job_ref.id),
                    "project": str(compile_job_ref.project.id),
                    "time_seconds": f"{end - start:.4f}",
                }
            }
        )
        result_refs = qnx.jobs.results(compile_job_ref)
        compiled_circuit_refs: list[CircuitRef | HUGRRef | QIRRef] = list()
        for ref in result_refs:
            assert isinstance(ref, CompilationResultRef)
            compiled_circuit_refs.append(ref.get_output())
        return NexusCompilationResult(
            compiled_circuit_refs,
            nexus_upload_result.benchmark_name,
            self.device,
            nexus_upload_result.nexus_project,
        )

    def run_compiled_circuits(
        self,
        compilation_result: NexusCompilationResult,
        shots_per_circuit: list[int],
        project: ProjectRef,
        jobname_suffix: str,
    ) -> list[dict[str, int]]:
        logger.info(
            f"Running circuits for benchmark {compilation_result.benchmark_name} on Nexus {self.device}"
        )
        start = time.perf_counter()
        compiled_circuit_refs = compilation_result.compiled_circuits
        execute_job_ref = qnx.start_execute_job(
            programs=compiled_circuit_refs,
            name=f"{compilation_result.benchmark_name}_execute_async-{jobname_suffix}",
            n_shots=shots_per_circuit,
            backend_config=self._nexus_device_config,
            project=project,
        )
        # Block until the job is complete
        qnx.jobs.wait_for(execute_job_ref)
        end = time.perf_counter()
        self.metrics.update(
            {
                "execute_job": {
                    "id": str(execute_job_ref.id),
                    "project": str(execute_job_ref.project.id),
                    "time_seconds": f"{end - start:.4f}",
                }
            }
        )
        # Retrieve a ExecutionResultRef for every Circuit that was run
        execute_job_result_refs = qnx.jobs.results(execute_job_ref)
        counts_per_circuit = list()
        for ref in execute_job_result_refs:
            assert isinstance(ref, ExecutionResultRef)
            result = ref.download_result()
            counts = result.get_counts(basis=BasisOrder.dlo)  # type: ignore
            # convert to dict[str, int] counts
            counts_per_circuit.append(
                {counter_key_to_string_key(k): v for k, v in counts.items()}
            )
        return counts_per_circuit

    def get_metrics(self) -> dict:
        return self.metrics
