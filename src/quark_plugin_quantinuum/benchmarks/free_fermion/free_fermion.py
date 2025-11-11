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
from typing import override, Dict
import logging

import matplotlib.pyplot as plt
import numpy as np
from quark.core import Core, Data, Result
from quark.interface_types import Other
from pytket.extensions.qiskit import qiskit_to_tk

from quark_plugin_quantinuum.interfaces.backend_result import BackendResult
from .free_fermion_helpers import (
    create_circuit,
    exact_values_and_variance,
    computes_score_values,
    extract_simulation_results,
)
from quark_plugin_quantinuum.interfaces.benchmark_circuits_pytket import (
    BenchmarkCircuitsPytket,
)

logger = logging.getLogger()


@dataclass
class FreeFermion(Core):
    """Implements the Free Fermion Simulation Benchmark

    Attributes:
        lx: Lattice width Lx used for the simulation. Must be a positive even integer.
        ly: Lattice height Ly used for the simulation. Must be a positive even integer.
        trotter_dt: Time step size
        trotter_n_step: Number of time steps (default: 2*Ly). Provide total number of steps as
                       integer if using custom value.
        create_plot: Whether to create a plot of the results (default: False)
        is used to show the plot dynamically.

    """

    lx: int = 2
    ly: int = 2
    trotter_dt: float = 0.2
    trotter_n_step: int | None = None
    create_plot: bool = False
    metrics: Dict[str, float | int | str] = field(init=False, default_factory=dict)

    # Validate fields and set default if necessary
    def __post_init__(self):
        for param, param_str in ((self.lx, "lx"), (self.ly, "ly")):
            if param < 2:
                raise ValueError(
                    f"{param_str} parameter must be >=2. Provided {param_str}: {param}"
                )
            if param % 2 != 0:
                raise ValueError(
                    f"{param_str} must be an even integer. Provided {param_str}: {param}"
                )
        # Set trotter_n_step if necessary
        # Also set self.internal_trotter_n_step for typing purposes
        #  -> has the same value and mypy knows it's not None
        if self.trotter_n_step is None:
            self.internal_trotter_n_step = 2 * self.ly
            self.trotter_n_step = 2 * self.ly
        else:
            self.internal_trotter_n_step = self.trotter_n_step

    @override
    def preprocess(self, data: None) -> Result:
        """
        Generate data that gets passed to the next submodule.

        :param data: This module requires no data. It generates a simulation
        :return: Input for a simulation backend
        """
        n_qubits = self.ly * self.lx * 3 // 2
        logger.info(
            f"Starting free fermion simulation benchmark on a {self.lx}x{self.ly} lattice ({n_qubits} qubits)"
        )
        logger.info(
            f"Using a trotter step size of {self.trotter_dt} and up to {self.internal_trotter_n_step} trotter steps"
        )
        circuits = [
            qiskit_to_tk(create_circuit(self.lx, self.ly, self.trotter_dt, n))
            for n in range(self.internal_trotter_n_step)
        ]
        return Data(Other(BenchmarkCircuitsPytket(circuits, self.benchmark_tag())))

    @override
    def postprocess(self, input_data: Other[BackendResult]) -> Result:
        """
        Processes data passed to this module from the submodule.

        :param input_data: The results of running the circuits on the backend
        :returns: A tuple containing the processed solution quality and the time taken for evaluation
        """

        backend_result = input_data.data

        counts_per_circuit, n_shots = backend_result.counts, backend_result.n_shots

        simulation_results = np.array(
            extract_simulation_results(
                self.trotter_dt, self.lx, self.ly, n_shots, counts_per_circuit
            )
        )
        exact_results = np.real(
            np.array(
                exact_values_and_variance(
                    self.internal_trotter_n_step, self.trotter_dt, self.lx, self.ly
                )
            )
        )
        score_gate, score_shot, score_runtime = computes_score_values(
            exact_results[:, 1] - simulation_results[:, 1],
            simulation_results[:, 2],
            exact_results[:, 2],
            self.lx * self.ly,
        )
        logger.info(f"Benchmark score (number of gates): {score_gate}")
        logger.info(f"Benchmark score (number of shots): {score_shot}")
        logger.info(f"Benchmark score (number of trotter steps): {score_runtime}")
        self.metrics.update(
            {
                "application_score_value": score_gate,
                "application_score_value_gates": score_gate,
                "application_score_value_shots": score_shot,
                "application_score_value_trotter_steps": score_runtime,
                "application_score_unit": "N_gates",
                "application_score_type": "int",
            }
        )
        if self.create_plot:
            self.create_and_handle_plot(
                simulation_results,
                exact_results,
                score_gate,
            )
        return Data(Other(score_gate))

    def get_metrics(self) -> dict:
        return self.metrics

    def create_and_handle_plot(
        self, simulation_results, exact_results, score_gates: int
    ) -> None:
        plt.plot(
            np.array(list(range(self.internal_trotter_n_step))) * self.trotter_dt,
            exact_results[:, 1],
            color="black",
            label="exact",
        )
        plt.errorbar(
            simulation_results[:, 0],
            simulation_results[:, 1],
            yerr=simulation_results[:, 2],
            label="simulated",
        )
        plt.title("SCORE = " + str(score_gates) + " gates")
        plt.xlabel("Time")
        plt.ylabel("Imbalance")
        plt.legend()
        plt.show()
        plt.close()

    def benchmark_tag(self) -> str:
        return f"free_fermion_{self.lx}_{self.ly}_{self.internal_trotter_n_step}_{self.trotter_dt}"
