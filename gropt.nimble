# GrOpt Nim bindings – Nimble project file

# Package
version     = "0.1.0"
author      = "GrOpt authors"
description = "Nim bindings for the GrOpt MRI gradient waveform optimisation library"
license     = "GPLv3"

# The library module is nim/gropt.nim (importable as `import gropt`).
# There is no installable binary; the package is a library.

# Dependencies
requires "nim >= 1.6.0"

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

task build_example, "Compile the example binary (nim/example)":
  exec "nim c -o:nim/example nim/example.nim"

task run_example, "Compile and run the example":
  exec "nim c -r nim/example.nim"

task build_release, "Compile the example binary with full optimisation":
  exec "nim c -d:release -o:nim/example nim/example.nim"
