"""Pytest configuration for emergence-lab test suite."""

import jax

# Enable persistent JIT compilation cache.
# First run compiles as usual; subsequent runs reuse cached XLA artifacts.
# Cache is automatically invalidated when function code, input shapes,
# JAX version, or hardware backend changes.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
