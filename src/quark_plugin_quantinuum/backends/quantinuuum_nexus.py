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
from typing import override
import logging

from pytket.circuit import BasisOrder
from qnexus.models.references import ExecutionResultRef
from quark.core import Core, Data, Result
from quark.interface_types import Other
import qnexus as qnx
import datetime

from .helpers import counter_key_to_string_key
from ..interfaces.nexus_compilation_result import NexusCompilationResult
from ..interfaces.backend_result import BackendResult

logger = logging.getLogger()


@dataclass
class QuantinuumNexus(Core):
    n_shots: int = 100
    _results: BackendResult | None = field(init=False, default=None)

    @override
    def preprocess(self, input_data: Other[NexusCompilationResult]) -> Result:
        backend_input = input_data.data
        compiled_circuits = backend_input.compiled_circuits
        benchmark_name = backend_input.benchmark_name
        nexus_project_name = backend_input.project_name
        device_name = backend_input.device_name
        project = qnx.projects.get_or_create(name=nexus_project_name)
        qnx.context.set_active_project(project)
        config = qnx.QuantinuumConfig(device_name=device_name)
        jobname_suffix = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")

        execute_job_ref = qnx.start_execute_job(
            programs=compiled_circuits,
            name=f"{benchmark_name}_execute_async-{jobname_suffix}",
            n_shots=[self.n_shots] * len(compiled_circuits),
            backend_config=config,
            project=project,
        )
        logger.info(
            f"Running circuits for benchmark {benchmark_name} on Nexus {device_name}"
        )
        # Block until the job is complete
        qnx.jobs.wait_for(execute_job_ref)
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

        self._results = BackendResult(counts=counts_per_circuit, n_shots=self.n_shots)
        return Data(None)

    @override
    def postprocess(self, input_data: Data) -> Result:
        return Data(Other(self._results))
