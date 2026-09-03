# XAD/QuantLibAAD integration note

The official QuantLib extensions page documents XAD/QuantLib as an external AAD integration. The QuantLibAAD repository describes a C++/CMake implementation that modifies QuantLib to use XAD tape-based AAD and optional JIT execution. The ordinary Python `QuantLib` wheel does not expose this native XAD tape automatically.

For a production native backend, build QuantLibAAD and XAD in a compatible C++ toolchain, then expose a narrow bridge to Python. The bridge should accept a validated pricing request, create AAD variables for all bumpable market handles, run the QuantLib payoff and Monte Carlo path logic on one tape, and return first- and second-order adjoints keyed by RFK. Keep a version/build identifier in the response.

Do not call `QuantLib.Option.delta()` or a finite-difference loop “XAD AAD” unless the loaded engine is the XAD-enabled QuantLibAAD build. The Python reverse-mode implementation in the companion project is a validation/reference backend for smooth payoffs. For discontinuous barrier and memory events, use native pathwise treatment or a declared event-risk decomposition; do not silently substitute finite differences when an AAD-only contract is requested.

References:

- https://www.quantlib.org/extensions.shtml
- https://github.com/auto-differentiation/QuantLibAAD
