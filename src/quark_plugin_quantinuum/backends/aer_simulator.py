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

from pytket.extensions.qiskit import tk_to_qiskit
from qiskit_aer import AerSimulator as QiskitAS
from qiskit import QuantumCircuit
from quark.core import Core, Data, Result
from quark.interface_types import Other

from ..interfaces.backend_result import BackendResult
from ..interfaces.benchmark_circuits_pytket import BenchmarkCircuitsPytket

logger = logging.getLogger()


@dataclass
class AerSimulator(Core):
    n_shots: int = 100
    _results: BackendResult | None = field(init=False, default=None)

    @override
    def preprocess(self, input_data: Other[BenchmarkCircuitsPytket]) -> Result:
        backend = QiskitAS()
        backend_input = input_data.data
        circuits = backend_input.circuits
        qiskit_circuits = [tk_to_qiskit(circ) for circ in circuits]
        self.warn_on_large_circuits(qiskit_circuits)

        counts_per_circuit = []
        logger.info(
            f"Running circuits for benchmark {backend_input.benchmark_name} on AerSimulator"
        )
        for n, circuit in enumerate(qiskit_circuits):
            logger.info(f"Running circuit for {n} Trotter steps")
            counts = (
                backend.run(circuit, shots=self.n_shots).result().get_counts(circuit)
            )
            counts_per_circuit.append(counts)

        self._results = BackendResult(counts=counts_per_circuit, n_shots=self.n_shots)
        return Data(None)

    @override
    def postprocess(self, input_data: Data) -> Result:
        return Data(Other(self._results))

    @staticmethod
    def warn_on_large_circuits(circuits: list[QuantumCircuit]) -> None:
        warning_n_qubits = 30
        max_n_qubit = max([circuit.num_qubits for circuit in circuits])
        if max_n_qubit > warning_n_qubits:
            logger.warning(
                f"Simulating circuits with over {warning_n_qubits} qubits. The high memory"
                f" requirements can lead to memory errors on some systems."
            )
