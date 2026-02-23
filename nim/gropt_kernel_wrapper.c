/*
 * gropt_kernel_wrapper.c
 *
 * Thin wrapper that pulls in the GrOpt optimisation kernel.
 * optimize_kernel.c contains a main() entry point intended for standalone
 * testing.  When linking with a Nim (or any other) executable the symbol
 * clashes with the host main().  Renaming it via the preprocessor before
 * the #include keeps the rest of the file intact while making the symbol
 * harmlessly unreachable dead code.
 *
 * The compiler will find optimize_kernel.c through the -I<srcDir> flag
 * that gropt.nim passes via {.passC.}.
 */
#define main gropt_main_unused
#include "optimize_kernel.c"
