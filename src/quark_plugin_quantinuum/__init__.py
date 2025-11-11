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
from quark_plugin_quantinuum.benchmarks.free_fermion.free_fermion import FreeFermion
from quark_plugin_quantinuum.miscellaneous.nexus_upload import NexusUpload

quark_modules = [
    ("free_fermion", FreeFermion),
    ("aer_simulator", AerSimulator),
    ("nexus_upload", NexusUpload),
    ("nexus_run", QuantinuumNexus),
]


def register() -> None:
    """
    Register all modules exposed to quark by this plugin.
    For each module, add a line of the form:
        factory.register("module_name", Module)

    The "module_name" will later be used to refer to the module in the configuration file.
    """
    for name, module in quark_modules:
        factory.register(name, module)


def print_available_quark_modules() -> None:
    print_data = [
        (name, f"{module.__module__}.{module.__qualname__}")
        for name, module in quark_modules
    ]
    # Determine column widths
    col1_width = max(len(row[0]) for row in print_data)
    col2_width = max(len(row[1]) for row in print_data)

    # Print header
    print(
        f"{'Module Name'.ljust(col1_width)} | {'Implementing Class'.ljust(col2_width)}"
    )
    print("-" * (col1_width + col2_width + 3))

    # Print rows
    for name, module in print_data:
        print(f"{name.ljust(col1_width)} | {module.ljust(col2_width)}")
