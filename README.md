# QUARK-plugin-quantinuum

This is a QUARK plugin that implements access to running QUARK benchmarks on Quantinuum devices.

Access is provided through [Quantinuum Nexus](https://nexus.quantinuum.com), and use of the Quantinuum backends
requires setting up a Nexus account. Please see the Nexus documentation for more information on the available devices.

Once you have a Nexus account, authentication is provided through the qnexus python package. With this plugin installed, simply run `qnx login` in the terminal before running your QUARK benchmark configuration.
