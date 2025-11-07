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

from pytket.extensions.qiskit import tk_to_qiskit
from quark.core import Core, Data, Result
from quark.interface_types import Other

from ..interfaces.backend_result import BackendResult
from ..interfaces.benchmark_circuits_pytket import BenchmarkCircuitsPytket
from ..interfaces.benchmark_circuits_qiskit import BenchmarkCircuitsQiskit

logger = logging.getLogger()


@dataclass
class PytketToQiskit(Core):
    """Converts benchmark circuits from pytket to qiskit circuit objects"""

    @override
    def preprocess(self, input_data: Other[BenchmarkCircuitsPytket]) -> Result:
        """
        Convert circuits from pytket to qiskit circuit objects
        """
        backend_input = input_data.data
        pytket_circuits = backend_input.circuits
        benchmark_name = backend_input.benchmark_name
        logger.info(
            f"Converting pytket circuits for benchmark {benchmark_name} to qiskit circuits"
        )
        qiskit_circuits = [tk_to_qiskit(circ) for circ in pytket_circuits]
        return Data(Other(BenchmarkCircuitsQiskit(qiskit_circuits, benchmark_name)))

    @override
    def postprocess(self, input_data: Other[BackendResult]) -> Result:
        """
        Pass results back to caller
        """
        return Data(input_data)
