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
from quark.plugin_manager import factory

from quark_plugin_quantinuum.backends.aer_simulator import AerSimulator
from quark_plugin_quantinuum.backends.quantinuuum_nexus import QuantinuumNexus
from quark_plugin_quantinuum.compilation.quantinuum_nexus import NexusCompilation
from quark_plugin_quantinuum.converters.pytket_to_qiskit_circuits import PytketToQiskit
from quark_plugin_quantinuum.free_fermion.free_fermion import FreeFermion


def register() -> None:
    """
    Register all modules exposed to quark by this plugin.
    For each module, add a line of the form:
        factory.register("module_name", Module)

    The "module_name" will later be used to refer to the module in the configuration file.
    """
    factory.register("free_fermion", FreeFermion)
    factory.register("aer_simulator", AerSimulator)
    factory.register("pytket_to_qiskit_circuits", PytketToQiskit)
    factory.register("nexus_compile", NexusCompilation)
    factory.register("nexus_run", QuantinuumNexus)
