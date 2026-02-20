## gropt – Nim bindings for the GrOpt MRI gradient waveform optimisation library.
##
## All C source files are compiled and linked **statically** into the final
## executable; no external shared library is required.
##
## The API mirrors the Python ``gropt(params)`` function.  The main entry
## point is the ``gropt`` proc which accepts a ``GrOptParams`` object and
## returns a ``GrOptResult``.
##
## Quick-start example::
##
##   import gropt
##
##   var p = initGrOptParams()
##   p.mode      = "diff_bval"
##   p.gmax      = 80.0     # mT/m  (values > 1.0 are treated as mT/m and
##                          #         converted to T/m internally)
##   p.smax      = 100.0    # T/m/s
##   p.TE        = 60.0     # ms
##   p.T_readout = 20.0     # ms
##   p.T_90      = 5.0      # ms
##   p.T_180     = 8.0      # ms
##   p.MMT       = 0        # null M0 only
##   p.dt        = 4e-4     # fixed time-step in seconds
##
##   let res = gropt(p, verbose = 1)
##   # res.G     – seq[seq[float]], shape [Naxis][N], values in T/m
##   # res.debug – seq[float] with 100 diagnostic values; [15] = runtime (s)

import std/[os, times]

# ---------------------------------------------------------------------------
# Compile-time path constants
# ---------------------------------------------------------------------------

const
  nimDir = currentSourcePath().parentDir()
  srcDir = nimDir / ".." / "src"

# ---------------------------------------------------------------------------
# Compiler / linker flags
# ---------------------------------------------------------------------------

{.passC: "-I" & srcDir.}   # find C headers in src/
{.passC: "-I" & nimDir.}   # find gropt_api.h in nim/
{.passC: "-std=c11 -O3".}
{.passL: "-lm".}

when defined(linux) or defined(bsd):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}
elif defined(macosx):
  # Requires libomp (e.g. `brew install libomp`)
  {.passC: "-Xpreprocessor -fopenmp".}
  {.passL: "-lomp".}

# ---------------------------------------------------------------------------
# Static C compilation
# The kernel wrapper renames optimize_kernel.c's main() to avoid a symbol
# clash with Nim's own main().
# ---------------------------------------------------------------------------

{.compile: srcDir / "cvx_matrix.c".}
{.compile: srcDir / "te_finder.c".}
{.compile: srcDir / "op_gradient.c".}
{.compile: srcDir / "op_maxwell.c".}
{.compile: srcDir / "op_bval.c".}
{.compile: srcDir / "op_beta.c".}
{.compile: srcDir / "op_eddy.c".}
{.compile: srcDir / "op_slewrate.c".}
{.compile: srcDir / "op_moments.c".}
{.compile: srcDir / "op_pns.c".}
{.compile: nimDir / "gropt_kernel_wrapper.c".}

# ---------------------------------------------------------------------------
# C function imports
# ---------------------------------------------------------------------------

proc c_free(p: pointer) {.importc: "free", header: "<stdlib.h>".}

proc c_run_kernel_diff_fixedN(
    G_out: ptr ptr cdouble, N_out: ptr cint, ddebug: ptr ptr cdouble,
    verbose: cint,
    N0: cint, gmax, smax, TE: cdouble,
    N_moments: cint, moments_params: ptr cdouble, PNS_thresh: cdouble,
    T_readout, T_90, T_180: cdouble, diffmode: cint, dt_out: cdouble,
    N_eddy: cint, eddy_params: ptr cdouble,
    search_bval, slew_reg: cdouble
) {.importc: "run_kernel_diff_fixedN", header: "gropt_api.h".}

proc c_run_kernel_diff_fixedN_Gin(
    G_out: ptr ptr cdouble, N_out: ptr cint, ddebug: ptr ptr cdouble,
    verbose: cint,
    G_in: ptr cdouble, N0: cint, gmax, smax, TE: cdouble,
    N_moments: cint, moments_params: ptr cdouble, PNS_thresh: cdouble,
    T_readout, T_90, T_180: cdouble, diffmode: cint, dt_out: cdouble,
    N_eddy: cint, eddy_params: ptr cdouble,
    search_bval, slew_reg: cdouble
) {.importc: "run_kernel_diff_fixedN_Gin", header: "gropt_api.h".}

proc c_run_kernel_diff_fixeddt(
    G_out: ptr ptr cdouble, N_out: ptr cint, ddebug: ptr ptr cdouble,
    verbose: cint,
    dt0, gmax, smax, TE: cdouble,
    N_moments: cint, moments_params: ptr cdouble, PNS_thresh: cdouble,
    T_readout, T_90, T_180: cdouble, diffmode: cint, dt_out: cdouble,
    N_eddy: cint, eddy_params: ptr cdouble,
    search_bval, slew_reg: cdouble,
    Naxis: cint
) {.importc: "run_kernel_diff_fixeddt", header: "gropt_api.h".}

proc c_run_kernel_diff_fixeddt_fixG(
    G_out: ptr ptr cdouble, N_out: ptr cint, ddebug: ptr ptr cdouble,
    verbose: cint,
    dt0, gmax, smax, TE: cdouble,
    N_moments: cint, moments_params: ptr cdouble, PNS_thresh: cdouble,
    T_readout, T_90, T_180: cdouble, diffmode: cint, dt_out: cdouble,
    N_eddy: cint, eddy_params: ptr cdouble, search_bval: cdouble,
    N_gfix: cint, gfix: ptr cdouble, slew_reg: cdouble
) {.importc: "run_kernel_diff_fixeddt_fixG", header: "gropt_api.h".}

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

type
  GrOptParams* = object
    ## Parameters for the GrOpt optimiser – mirrors the Python ``params`` dict.

    mode*: string
      ## Optimisation mode.  One of:
      ##   ``"diff_bval"``  – b-value constrained diffusion (diffmode 2)
      ##   ``"diff_beta"``  – beta/moment constrained diffusion (diffmode 1)
      ##   ``"free"``       – free gradient design with user-supplied moments

    gmax*: float
      ## Maximum gradient amplitude.
      ## Values **> 1.0** are treated as **mT/m** and converted to T/m
      ## (divided by 1000) before passing to the solver.
      ## Values **<= 1.0** are treated as **T/m** directly.

    smax*: float
      ## Maximum slew rate in **T/m/s**.

    TE*: float
      ## Echo time in **ms**.

    T_90*: float
      ## Duration of the 90° RF pulse in **ms**.
      ## Required for ``"diff_bval"`` and ``"diff_beta"`` modes.

    T_180*: float
      ## Duration of the 180° RF pulse in **ms**.
      ## Required for ``"diff_bval"`` and ``"diff_beta"`` modes.

    T_readout*: float
      ## Readout duration in **ms**.
      ## Required for ``"diff_bval"`` and ``"diff_beta"`` modes.

    MMT*: int
      ## Moment nulling order for diffusion modes.
      ##   0 → null M0 only
      ##   1 → null M0 and M1
      ##   2 → null M0, M1, and M2

    tol*: float
      ## Moment constraint tolerance (default ``1e-3``).

    N0*: int
      ## Fixed number of gradient samples.
      ## Set to a positive value to use the fixed-N kernel.
      ## Use ``-1`` to specify the time step via ``dt`` instead.

    dt*: float
      ## Fixed time step in **seconds**.
      ## Set to a positive value to use the fixed-dt kernel.
      ## Use ``-1.0`` to specify the sample count via ``N0`` instead.

    dt_out*: float
      ## Output time step for resampling in **seconds**.
      ## Set to ``-1.0`` to return the native time step.

    eddy_params*: seq[float]
      ## Eddy-current constraint parameters, given as a flat sequence in
      ## groups of 4: ``[lambda_ms, goal, tolerance, offset]`` per constraint.
      ## Leave empty to disable eddy-current constraints.

    pns_thresh*: float
      ## Peripheral nerve stimulation threshold.
      ## Set to ``-1.0`` to disable PNS constraints.

    gfix*: seq[float]
      ## Fixed gradient values (flat, length = ``N * Naxis``).
      ## Leave empty to disable fixed-gradient constraints.

    slew_reg*: float
      ## Slew-rate regularisation weight (default ``1.0``).

    Naxis*: int
      ## Number of gradient axes (default ``1``).
      ## Only used with the fixed-dt kernel.

    moment_params*: seq[float]
      ## Custom moment constraint parameters for ``"free"`` mode.
      ## Flat sequence in groups of 7:
      ##   ``[order, moment_idx, target, lower_limit, upper_limit, offset, tolerance]``
      ## per constraint.

  GrOptResult* = object
    ## Optimised gradient waveform and diagnostics.

    G*: seq[seq[float]]
      ## Gradient waveform, shape ``[Naxis][N]``, values in **T/m**.

    debug*: seq[float]
      ## 100-element diagnostic array.
      ## Index 15 holds the wall-clock runtime in seconds (set by the binding,
      ## mirroring the Python wrapper behaviour).

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

proc initGrOptParams*(): GrOptParams =
  ## Return a ``GrOptParams`` object pre-filled with sensible defaults.
  ##
  ## Override the fields you need before passing to ``gropt()``.
  result = GrOptParams(
    mode:        "diff_bval",
    gmax:        0.08,   # 80 mT/m expressed as T/m (≤ 1.0 → no conversion)
    smax:        100.0,
    TE:          60.0,
    T_90:        5.0,
    T_180:       8.0,
    T_readout:   20.0,
    MMT:         0,
    tol:         1.0e-3,
    N0:          -1,
    dt:          -1.0,
    dt_out:      -1.0,
    pns_thresh:  -1.0,
    slew_reg:    1.0,
    Naxis:       1,
  )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

proc buildMomentFlat(params: GrOptParams, diffmode: int): (seq[cdouble], int) =
  ## Build the flat C moment_params array and the moment count.
  ##
  ## For diffusion modes the array is auto-constructed from ``params.MMT``.
  ## For free mode ``params.moment_params`` is used directly (groups of 7).
  var flat: seq[cdouble]
  var count: int

  if diffmode > 0:
    let tol = params.tol.cdouble
    # Each row: [order=0, moment_idx, target=0, lower=-1, upper=-1, offset=0, tol]
    flat.add(0.cdouble); flat.add(0.cdouble); flat.add(0.cdouble)
    flat.add(-1.cdouble); flat.add(-1.cdouble); flat.add(0.cdouble); flat.add(tol)
    count = 1
    if params.MMT > 0:
      flat.add(0.cdouble); flat.add(1.cdouble); flat.add(0.cdouble)
      flat.add(-1.cdouble); flat.add(-1.cdouble); flat.add(0.cdouble); flat.add(tol)
      count = 2
    if params.MMT > 1:
      flat.add(0.cdouble); flat.add(2.cdouble); flat.add(0.cdouble)
      flat.add(-1.cdouble); flat.add(-1.cdouble); flat.add(0.cdouble); flat.add(tol)
      count = 3
  else:
    for v in params.moment_params:
      flat.add(v.cdouble)
    count = params.moment_params.len div 7

  # Guard: the C code dereferences the pointer even when count == 0
  if flat.len == 0:
    flat.add(0.cdouble)

  result = (flat, count)

proc toCDoubleSeq(s: seq[float]): seq[cdouble] =
  ## Convert seq[float] to seq[cdouble], returning a one-element dummy when empty.
  if s.len > 0:
    for v in s:
      result.add(v.cdouble)
  else:
    result.add(0.cdouble)

# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

proc gropt*(params: GrOptParams, verbose: int = 0): GrOptResult =
  ## Optimise an MRI gradient waveform and return the result.
  ##
  ## The correct C kernel is selected automatically:
  ##
  ## * ``params.N0 > 0``  → fixed-N kernel (``run_kernel_diff_fixedN``)
  ## * ``params.dt > 0`` and ``params.gfix`` non-empty
  ##   → fixed-dt kernel with fixed-gradient constraints
  ## * ``params.dt > 0`` and ``params.gfix`` empty
  ##   → fixed-dt kernel (``run_kernel_diff_fixeddt``)
  ##
  ## **Return value:**
  ## ``result.G`` has shape ``[Naxis][N]`` with values in T/m.
  ## ``result.debug[15]`` is the wall-clock runtime in seconds.

  let diffmode = case params.mode
    of "diff_bval": 2
    of "diff_beta": 1
    of "free":      0
    else:
      raise newException(ValueError,
        "gropt: invalid mode '" & params.mode &
        "'; expected \"diff_bval\", \"diff_beta\", or \"free\"")

  # Scale gmax: the Python wrapper divides by 1000 when the value exceeds 1.0
  var gmax = params.gmax
  if gmax > 1.0:
    gmax /= 1000.0

  # Timing / RF durations (zeroed out for free mode, matching Python behaviour)
  let T_readout = cdouble(if diffmode > 0: params.T_readout else: 0.0)
  let T_90      = cdouble(if diffmode > 0: params.T_90      else: 0.0)
  let T_180     = cdouble(if diffmode > 0: params.T_180     else: 0.0)

  # Build parameter arrays
  var (moment_flat, N_moments) = buildMomentFlat(params, diffmode)
  var eddy_flat = toCDoubleSeq(params.eddy_params)
  var gfix_flat = toCDoubleSeq(params.gfix)

  let N_eddy = cint(params.eddy_params.len div 4)
  let N_gfix = cint(params.gfix.len)

  # Output pointers – the C functions malloc these; we dealloc after copying
  var G_out_ptr:  ptr cdouble = nil
  var ddebug_ptr: ptr cdouble = nil
  var N_out:      cint        = 0

  let t0 = getTime()

  if params.N0 > 0:
    c_run_kernel_diff_fixedN(
      addr G_out_ptr, addr N_out, addr ddebug_ptr, cint(verbose),
      cint(params.N0), cdouble(gmax), cdouble(params.smax), cdouble(params.TE),
      cint(N_moments), addr moment_flat[0], cdouble(params.pns_thresh),
      T_readout, T_90, T_180, cint(diffmode), cdouble(params.dt_out),
      N_eddy, addr eddy_flat[0], -1.0, cdouble(params.slew_reg))

  elif params.dt > 0.0:
    if N_gfix > 0:
      c_run_kernel_diff_fixeddt_fixG(
        addr G_out_ptr, addr N_out, addr ddebug_ptr, cint(verbose),
        cdouble(params.dt), cdouble(gmax), cdouble(params.smax), cdouble(params.TE),
        cint(N_moments), addr moment_flat[0], cdouble(params.pns_thresh),
        T_readout, T_90, T_180, cint(diffmode), cdouble(params.dt_out),
        N_eddy, addr eddy_flat[0], -1.0,
        N_gfix, addr gfix_flat[0], cdouble(params.slew_reg))
    else:
      c_run_kernel_diff_fixeddt(
        addr G_out_ptr, addr N_out, addr ddebug_ptr, cint(verbose),
        cdouble(params.dt), cdouble(gmax), cdouble(params.smax), cdouble(params.TE),
        cint(N_moments), addr moment_flat[0], cdouble(params.pns_thresh),
        T_readout, T_90, T_180, cint(diffmode), cdouble(params.dt_out),
        N_eddy, addr eddy_flat[0], -1.0, cdouble(params.slew_reg),
        cint(params.Naxis))

  else:
    raise newException(ValueError,
      "gropt: must set params.N0 > 0 or params.dt > 0.0")

  let elapsed = float((getTime() - t0).inMicroseconds) / 1_000_000.0

  # Copy gradient (C layout is axis-major: all samples for axis 0, then axis 1, …)
  let N = int(N_out) div params.Naxis
  let G_arr     = cast[ptr UncheckedArray[cdouble]](G_out_ptr)
  let debug_arr = cast[ptr UncheckedArray[cdouble]](ddebug_ptr)

  result.G = newSeq[seq[float]](params.Naxis)
  for ax in 0 ..< params.Naxis:
    result.G[ax] = newSeq[float](N)
    for i in 0 ..< N:
      result.G[ax][i] = float(G_arr[ax * N + i])

  # Copy diagnostics
  result.debug = newSeq[float](100)
  for i in 0 ..< 100:
    result.debug[i] = float(debug_arr[i])
  result.debug[15] = elapsed   # mirror Python: debug[15] = wall-clock time (s)

  # Release C-allocated memory (malloc'd by the kernel; Nim's dealloc wraps free)
  c_free(G_out_ptr)
  c_free(ddebug_ptr)


proc groptWithGin*(Gin: openArray[float], params: GrOptParams,
                   verbose: int = 0): GrOptResult =
  ## Optimise using an initial gradient waveform ``Gin``.
  ##
  ## Uses the fixed-N kernel (``run_kernel_diff_fixedN_Gin``).
  ## ``params.N0`` must be set to a positive value.
  ## ``Gin`` must contain exactly ``N0`` elements.
  ##
  ## This mirrors the Python ``run_diffkernel_fixN_Gin`` helper.
  if params.N0 <= 0:
    raise newException(ValueError,
      "groptWithGin: params.N0 must be > 0")
  if Gin.len != params.N0:
    raise newException(ValueError,
      "groptWithGin: Gin length (" & $Gin.len &
      ") must equal params.N0 (" & $params.N0 & ")")

  let diffmode = case params.mode
    of "diff_bval": 2
    of "diff_beta": 1
    of "free":      0
    else:
      raise newException(ValueError,
        "groptWithGin: invalid mode '" & params.mode & "'")

  var gmax = params.gmax
  if gmax > 1.0:
    gmax /= 1000.0

  let T_readout = cdouble(if diffmode > 0: params.T_readout else: 0.0)
  let T_90      = cdouble(if diffmode > 0: params.T_90      else: 0.0)
  let T_180     = cdouble(if diffmode > 0: params.T_180     else: 0.0)

  var (moment_flat, N_moments) = buildMomentFlat(params, diffmode)
  var eddy_flat = toCDoubleSeq(params.eddy_params)
  let N_eddy = cint(params.eddy_params.len div 4)

  # Copy input gradient into a C-compatible buffer
  var G_in_c = newSeq[cdouble](Gin.len)
  for i, v in Gin:
    G_in_c[i] = cdouble(v)

  var G_out_ptr:  ptr cdouble = nil
  var ddebug_ptr: ptr cdouble = nil
  var N_out:      cint        = 0

  let t0 = getTime()

  c_run_kernel_diff_fixedN_Gin(
    addr G_out_ptr, addr N_out, addr ddebug_ptr, cint(verbose),
    addr G_in_c[0], cint(params.N0),
    cdouble(gmax), cdouble(params.smax), cdouble(params.TE),
    cint(N_moments), addr moment_flat[0], cdouble(params.pns_thresh),
    T_readout, T_90, T_180, cint(diffmode), cdouble(params.dt_out),
    N_eddy, addr eddy_flat[0], -1.0, cdouble(params.slew_reg))

  let elapsed = float((getTime() - t0).inMicroseconds) / 1_000_000.0

  let N = int(N_out)
  let G_arr     = cast[ptr UncheckedArray[cdouble]](G_out_ptr)
  let debug_arr = cast[ptr UncheckedArray[cdouble]](ddebug_ptr)

  result.G = @[newSeq[float](N)]
  for i in 0 ..< N:
    result.G[0][i] = float(G_arr[i])

  result.debug = newSeq[float](100)
  for i in 0 ..< 100:
    result.debug[i] = float(debug_arr[i])
  result.debug[15] = elapsed

  c_free(G_out_ptr)
  c_free(ddebug_ptr)
