## example.nim – demonstrates the GrOpt Nim bindings.
##
## Mirrors the canonical Python usage:
##
##   params = {}
##   params['mode']      = 'diff_bval'
##   params['gmax']      = 80       # mT/m
##   params['smax']      = 100      # T/m/s
##   params['TE']        = 60       # ms
##   params['T_readout'] = 20
##   params['T_90']      = 5
##   params['T_180']     = 8
##   params['MMT']       = 0
##   params['dt']        = 4e-4
##   G, ddebug = gropt.gropt(params)
##
## Build and run:
##   cd nim
##   nim c -r example.nim
##
## Or via Nimble:
##   nimble run_example

import gropt
import std/[math, strformat, strutils]

# ---------------------------------------------------------------------------
# Example 1: diffusion gradient (diff_bval mode, fixed dt)
# ---------------------------------------------------------------------------
proc exampleDiffBval() =
  echo "=".repeat(60)
  echo "Example 1: diff_bval mode, fixed dt"
  echo "=".repeat(60)

  var p = initGrOptParams()
  p.mode      = "diff_bval"
  p.gmax      = 80.0    # mT/m
  p.smax      = 100.0   # T/m/s
  p.TE        = 60.0    # ms
  p.T_readout = 20.0    # ms
  p.T_90      = 5.0     # ms
  p.T_180     = 8.0     # ms
  p.MMT       = 0
  p.dt        = 4e-4    # seconds

  let res = gropt(p, verbose = 0)

  let Naxis = res.G.len
  let N     = if Naxis > 0: res.G[0].len else: 0
  echo &"  Gradient shape : {Naxis} axis × {N} samples"
  echo &"  Runtime        : {res.debug[15]:.3f} s"
  echo &"  Iterations     : {res.debug[0]:.0f}"

  if N > 0:
    var gmaxActual = 0.0
    for g in res.G[0]:
      gmaxActual = max(gmaxActual, abs(g))
    echo &"  Max gradient   : {gmaxActual * 1000:.2f} mT/m  (limit {p.gmax:.0f} mT/m)"

# ---------------------------------------------------------------------------
# Example 2: diffusion gradient (diff_bval mode, fixed N0, M0+M1 nulling)
# ---------------------------------------------------------------------------
proc exampleDiffBvalFixedN() =
  echo ""
  echo "=".repeat(60)
  echo "Example 2: diff_bval mode, fixed N0, MMT=1"
  echo "=".repeat(60)

  var p = initGrOptParams()
  p.mode      = "diff_bval"
  p.gmax      = 80.0
  p.smax      = 100.0
  p.TE        = 60.0
  p.T_readout = 20.0
  p.T_90      = 5.0
  p.T_180     = 8.0
  p.MMT       = 1        # null M0 and M1
  p.N0        = 100      # fixed sample count

  let res = gropt(p, verbose = 0)

  let N = if res.G.len > 0: res.G[0].len else: 0
  echo &"  Gradient shape : 1 axis × {N} samples"
  echo &"  Runtime        : {res.debug[15]:.3f} s"
  echo &"  Iterations     : {res.debug[0]:.0f}"

  if N > 0:
    var gmaxActual = 0.0
    for g in res.G[0]:
      gmaxActual = max(gmaxActual, abs(g))
    echo &"  Max gradient   : {gmaxActual * 1000:.2f} mT/m"

# ---------------------------------------------------------------------------
# Example 3: free-mode gradient design with custom moment constraints
# ---------------------------------------------------------------------------
proc exampleFree() =
  echo ""
  echo "=".repeat(60)
  echo "Example 3: free mode with moment constraints (flow comp)"
  echo "=".repeat(60)

  var p = initGrOptParams()
  p.mode = "free"
  p.gmax = 50.0         # mT/m
  p.smax = 150.0        # T/m/s
  p.TE   = 10.0         # ms
  p.dt   = 1e-4         # seconds

  # Moment constraints: [order, moment_idx, target, lower, upper, offset, tol]
  # Null M0 and M1 (flow compensation)
  p.moment_params = @[
    0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3,   # M0 = 0
    0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0e-3,   # M1 = 0
  ]

  let res = gropt(p, verbose = 0)

  let N = if res.G.len > 0: res.G[0].len else: 0
  echo &"  Gradient shape : 1 axis × {N} samples"
  echo &"  Runtime        : {res.debug[15]:.3f} s"
  echo &"  Iterations     : {res.debug[0]:.0f}"

  if N > 0:
    var gmaxActual = 0.0
    for g in res.G[0]:
      gmaxActual = max(gmaxActual, abs(g))
    echo &"  Max gradient   : {gmaxActual * 1000:.2f} mT/m  (limit {p.gmax:.0f} mT/m)"

# ---------------------------------------------------------------------------
# Example 4: multi-axis gradient (Naxis = 3)
# ---------------------------------------------------------------------------
proc exampleMultiAxis() =
  echo ""
  echo "=".repeat(60)
  echo "Example 4: 3-axis diffusion gradient"
  echo "=".repeat(60)

  var p = initGrOptParams()
  p.mode      = "diff_bval"
  p.gmax      = 80.0
  p.smax      = 100.0
  p.TE        = 80.0
  p.T_readout = 20.0
  p.T_90      = 5.0
  p.T_180     = 8.0
  p.MMT       = 0
  p.dt        = 4e-4
  p.Naxis     = 3

  let res = gropt(p, verbose = 0)

  let Naxis = res.G.len
  let N     = if Naxis > 0: res.G[0].len else: 0
  echo &"  Gradient shape : {Naxis} axes × {N} samples"
  echo &"  Runtime        : {res.debug[15]:.3f} s"
  echo &"  Iterations     : {res.debug[0]:.0f}"

  for ax in 0 ..< Naxis:
    var gmaxAx = 0.0
    for g in res.G[ax]:
      gmaxAx = max(gmaxAx, abs(g))
    echo &"  Axis {ax}: max gradient = {gmaxAx * 1000:.2f} mT/m"

# ---------------------------------------------------------------------------
# Example 5: groptWithGin – warm-start from a previous solution
# ---------------------------------------------------------------------------
proc exampleWithGin() =
  echo ""
  echo "=".repeat(60)
  echo "Example 5: warm-start with groptWithGin"
  echo "=".repeat(60)

  var p = initGrOptParams()
  p.mode      = "diff_bval"
  p.gmax      = 80.0
  p.smax      = 100.0
  p.TE        = 60.0
  p.T_readout = 20.0
  p.T_90      = 5.0
  p.T_180     = 8.0
  p.MMT       = 0
  p.N0        = 80

  # First pass: cold start
  let res0 = gropt(p, verbose = 0)
  echo &"  Cold-start iterations : {res0.debug[0]:.0f}  ({res0.debug[15]:.3f} s)"

  # Second pass: warm-start from the first solution
  let res1 = groptWithGin(res0.G[0], p, verbose = 0)
  echo &"  Warm-start iterations : {res1.debug[0]:.0f}  ({res1.debug[15]:.3f} s)"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
exampleDiffBval()
exampleDiffBvalFixedN()
exampleFree()
exampleMultiAxis()
exampleWithGin()
echo ""
echo "Done."
