#!/usr/bin/env python3
import argparse
import sys
import math


def sum_bc(n: int, k: int):
    """
    return the sum of the binomial coefficient
    """
    return sum([math.comb(n, i) for i in range(k+1)])


# return the index of the coefficient c_ij in grevlex order
# (c_01 has index 0), assume i < j
def COEF(i, j):
    return ((j * (j - 1)) >> 1) + i


def bit_to_idx(b, n):
    """
    convert 01000 into 3, etc
    """
    for i in range(n):
        if (b >> i) == 1:
            return i

    return 0


parser = argparse.ArgumentParser(
    description="Generate GPU kernel for fast exhaustive search.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-u",
    "--unroll",
    dest="unroll",
    type=int,
    required=True,
    help="level to unroll code (>0)",
)

parser.add_argument(
    "-d",
    "--deg2-unroll",
    dest="deg2unroll",
    type=int,
    default=0,
    help="level to deg2 unroll code (>0)",
)

parser.add_argument(
    "-t",
    "--type",
    dest="type",
    action="store_true",
    help="if set `uint64_t` are used internally",
)


args = parser.parse_args()
unroll = args.unroll
data_type = "uint32_t" if not args.type else "uint64_t"
restricted = False  # TODO
only_code = False
deg2unroll = args.deg2unroll


if not unroll > 0:
    sys.stderr.write('"unroll" must be at least 1!\n')
    sys.exit(-1)

if unroll > 32:
    print("more than 2**32 loops not supported")
    sys.exit(-2)


if not only_code:
    print("#include <assert.h>")
    print("#include <stdint.h>")
    print()

    print("#if !defined(ASSERT)")
    print("#if defined(DEBUG)\n#define ASSERT(x) assert(x)\n"
          "#else\n#define ASSERT(x)\n#endif")
    print("#endif")
    print()
    print("#if !defined(COEF)\n#define COEF(I,J) ((((J)*((J)-1))>>1) + (I))\n#endif")
    print()
    print("#if !defined(idx)\n#define idx (blockIdx.x*blockDim.x + threadIdx.x)\n#endif")
    print()

    print("#if !defined(KERNEL_SOLUTIONS)")
    print('#error "kernel only works if `KERNEL_SOLUTIONS` is defined"')
    print("#endif")
    print()

    print("#if !defined(KERNEL_SHARED_SOLUTIONS)")
    print('#error "kernel only works if `KERNEL_SHARED_SOLUTIONS` is defined"')
    print("#endif")
    print()

    print("// MAX_K is the maxium number of variables this kernel can bruteforce.")
    print("// This limiation is only needed to reduce the max mem/thread, needed")
    print("// for the differentials.")
    print("// Second its needed to reduce the memory needed to save the quadratic")
    print("// parts of the equations system.")
    print("// The maximum for MAX_K is 31 or 63, depending on T, but we always")
    print("// need the highest bit to indicate we found multiple solutions.")
    print("#if !defined(MAX_K)\n#define MAX_K 20\n#endif")
    print()
    print("#if !defined(BLOCK_DIM)")
    print('#error "BLOCK_DIM not defined"')
    print("#endif")
    print()

    # print some information about the kernel
    print(
        """
/// NOTE: let in the following n = #varibales, and k = #variables to bruteforce
///       so in our case we want to set k on n1.
/// If k = n1, this means that we need 2**(n-k) threads at most do run this
/// code. Note that this is only true if we do not have any weight restriction.
/// \\param deg1 lineare terms of the form
///   [
////     <---------------------2**(N-K)---------------------->
///       fm,....f0
/// |   [[x0,....x0]*Pk(0), ..., [x0, ..., x0]*Pk(2**(n-k) -1)]
///      <-  T  ->               <--  T    -->
/// k+1           ...
/// |   [[x_{k-1},...x_{k-1}]*Pk(0),...,[x_{k-1},...,x_{k-1}]*Pk(2**(n-k)-1)]
///     [b,....b]
///   ]
/// where Pk(y) is the polynomial only containing variables x_k,..x_(n-1)
/// evaluated at y.
/// IMPORTANT NOTE: If you look at the code of `libFES` and how they enumerate
///     partial solutions and push them into a kernel, you will see, that
///     they push basically columns of the above picture into the kernel.
///
/// \\param deg2_block: quadratic part of the polyomials:
/// 		[
///   		 <- 						N(N-1)/2 							   		->
///   		 <-  32 bit     ->						   <- 		  32bit	      		->
///	  		[[x0x1, ..., x0x1], [x0x2,...,x0x2], ..., [x_{n-1}xn, ..., x_{n-2}x_{n-1}]
///   		   fm         f1	  fm 	   f1 			fm					 f1
///         ]
/// \\param result return values of the form:
/// tid: 0     1    2          2**(n-k)-1
///     [sol, sol, sol, ...., sol]
/// NOTE: if `KERNEL_SOLUTIONS` is defined > 0 than the functionality of the
///     kernel changes. Instead of beeing able to find atmost one solution,
///     each kernel can find up to `KERNEL_SOLUTIONS` many solutions, which are
///     saved in result[KERNEL_SOLUTIONS*idx + i] for i in range(KERNEL_SOLUTIONS).
/// \\param num_threads number of threads, e.g. tpb * blocks,
///                 should be 2**(n-k) or |W_{n-n1}^{w}|
/// \\param k number of variables to bruteforce: n-n1"""
    )

    # start of the kernel function
    print("template<const uint32_t tpb=1024>")
    print(
        "__global__ void guess(const uint32_t *deg1, uint32_t *result, uint32_t num_threads, const uint32_t k)"
    )
    print("{")
    print()

# start with the content of the function
print("   __shared__ {0} shared_solutions[tpb*KERNEL_SHARED_SOLUTIONS];".format(data_type))
print()
print("  {0} x = {1}; // for round zero".format(data_type, 1 << (unroll - 1)))
print("  {0} y = 0;".format(data_type))
print("  {0} z = 0;".format(data_type))
print("  const {0} off = threadIdx.x * KERNEL_SHARED_SOLUTIONS;".format(data_type))
# print("  {0} sol = 0;".format(data_type))
print("  float sol_cnt = 0.0;")
if not only_code:
    # id only code is printed, this must be defined outside
    print("  float total_sol_cnt = 0.0;")
print("  {0} block = 0;".format(data_type))
print("  {0} tmp = 0;".format(data_type))
print("constexpr uint32_t blockk = 0;")
print("")
print("  constexpr {0} unroll = {1};".format(data_type, unroll))
print("  assert(unroll < k);")
# print("  ASSERT(k < MAX_K);")
print("")

# print("// clearing mem")
# print("for (uint32_t i = 0; i < KERNEL_SOLUTIONS; i++) {")
# print("     result[idx*KERNEL_SOLUTIONS + i] = 0;")
# print("}")
print("")

# initialize diff and res (we can predict when a bit is first flipped)
print("  {0} diff0 = deg1[num_threads * 0 + idx + blockk];".format(data_type))

for i in range(1, unroll):
    print("  {1} diff{0} = ".format(i, data_type), end="")
    print("deg1[num_threads * {0} + idx + blockk] ^ deg2_block[{1}] ^ deg2_block[{0}];".format(
            i, COEF(i - 1, i)
        )
    )

print("")
print("// memory for `k`-`unroll` differentials")
print("  {0} diff[MAX_K];".format(data_type))
print("")

# start loop
print("  for (uint32_t i = {0}; i < k; i++)".format(unroll))
print(
    "    diff[i-{0}] = deg1[num_threads * i + idx + blockk] ^ deg2_block[COEF(i-1, i)] ^ deg2_block[i];".format(
        unroll
    )
)

print("")
print("  // undo changes of first round")
print("  {0} res = deg1[num_threads * k + idx + blockk] ^ diff0 ^ deg2_block[0];".format(data_type))
print("")

if deg2unroll > 0:
    print()
    print("  // memory for the 'internal_d'- block2 values")
    for i in range(deg2unroll):
        print("  const {1} internal_deg2_{0} = deg2_block[{0}];".format(i, data_type))


print("  __syncthreads();")
print("")

# additional check
if restricted:
    print("    if (idx < sum_bc(uint64_t(n-k), k)) {")

# main loop
print(
    "  for (uint32_t rounds = 0; rounds < (1 << k); rounds += (1 << {0})) ".format(
        unroll
    )
)

# this basically computes the first round
print("    {")
print("    tmp = (rounds & (rounds-1));")
print("    y = rounds ^ tmp;")
print("    x ^= (y ^ {0});".format(1 << (unroll - 1)))  # important!
print("    z = tmp ^ (tmp & (tmp-1));")

print("")
print("    {0} y_pos = y == 0 ? 0 : __ffs(y) - 1;".format(data_type))
print("    {0} z_pos = z == 0 ? 0 : __ffs(z) - 1;".format(data_type))
# print("    uint32_t y_pos = ctz(y);")
# print("    uint32_t z_pos = ctz(z);")
print("")
print("    block = y_pos * (y_pos-1) / 2;")
print("")
print("    if (y_pos == 0) {")
print("      diff0 ^= deg2_block[COEF(y_pos, z_pos)];")
print("      res ^= diff0;")
print("      //ASSERT(diff0 != 0);")
print("    } else {")
print("      diff[y_pos - {0}] ^= deg2_block[COEF(y_pos, z_pos)];".format(unroll))
print("      res ^= diff[y_pos - {0}];".format(unroll))
print("      //ASSERT(diff[y_pos-1]!= 0);")
print("    }")

# check for the first solition
print("")
print("    if (res == 0) shared_solutions[off + static_cast<uint32_t>(sol_cnt)] = x;")
print("    if (res == 0) sol_cnt += 1.0;")
print("")

print("    // start unrolled loop")

x = 0

for i in range(1, 1 << unroll):
    y = i ^ (i & (i - 1))
    x ^= y
    tmp = y ^ i
    z = tmp ^ (tmp & (tmp - 1))

    y = bit_to_idx(y, unroll)
    z = bit_to_idx(z, unroll)

    if i > 1:
        print("")

    if z == 0:  # first flip in block
        print("    diff{0} ^= deg2_block[block++];".format(y))
    else:
        if deg2unroll > 0 and deg2unroll > COEF(y, z):
            # print("    diff{0} ^= internal_block2[{1}];".format(y, COEF(y, z)))
            print("    diff{0} ^= internal_deg2_{1};".format(y, COEF(y, z)))
        else:
            print("    diff{0} ^= deg2_block[{1}];".format(y, COEF(y, z)))

    print("    res ^= diff{0};".format(y))

    print("    if ((float)res == 0.0) shared_solutions[off + static_cast<uint32_t>(sol_cnt)] = {0} ^ x;".format(x))
    print("    if ((float)res == 0.0) sol_cnt += 1.0;")


print("    // end unrolled loop")

# if the `solutions` was specified we write back every loop iterations
# all possible solutions back
print()
print("    ASSERT(static_cast<uint32_t>(sol_cnt) < KERNEL_SHARED_SOLUTIONS);")
print()
print("    // write back the solutions.")
print("    for(uint8_t i = 0; i < static_cast<uint32_t> (sol_cnt); i++) {")
print(r'    	//printf("tid: %d, round: %d, found %f sol: %d\n", idx, rounds, sol_cnt, shared_solutions[off + i]);')
print("		result[idx*KERNEL_SOLUTIONS + static_cast<uint32_t>(total_sol_cnt) + i] = shared_solutions[off + i];")
print("	}")
print()
print("	total_sol_cnt += sol_cnt;")
print("	sol_cnt = 0.;")
print()
print("  ASSERT(static_cast<uint32_t>(total_sol_cnt) < KERNEL_SOLUTIONS);")
print()

# closing bracked for the main unrolled loop
print("    } // end of main loop")
print()

# closing bracked if we are in restricted setting
if restricted:
    print("  }")
print("")


if not only_code:
    # closing bracked for the whole function/kernel
    print("}")
    print("")
