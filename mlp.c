/*
 * mlp.c - Multi-Layer Perceptron Core for CC65 / Atari 8-bit
 *
 * PURPOSE
 * -------
 * This file implements the core neural network operations. It defines a 2-4-1
 * Multi-Layer Perceptron trained to learn the XOR function.  It is a C port of 
 * the original XORTRAN.FOR Fortran program, adapted for the CC65 cross-compiler 
 * targeting the Atari 8-bit family (6502 processor).
 *
 * All arithmetic uses Q8.8 fixed-point format (16-bit signed, value * 256).
 *
 * MODULES
 * -------
 *   Global data:
 *     sig_table[]  - Sigmoid lookup table (33 entries, x from -8.0 to +8.0)
 *     data_common  - COMMON /DATA/ (INPUTS, TARGETS)
 *     wghts_common - COMMON /WGHTS/ (W1, B1, W2, B2)
 *     consts_common- COMMON /CONSTS/ (EXPLIM, LR)
 *     layer_common - COMMON /LAYER/ (Z1, A1, Z2, A2)
 *
 *   Activation functions:
 *     sigmoi()     - Sigmoid via lookup table with linear interpolation
 *     sigdrv()     - Sigmoid derivative: s * (1 - s)
 *     tanhac()     - Tanh activation (stub, returns input unchanged)
 *     tanhdr()     - Tanh derivative (stub, returns 1.0)
 *     relu()       - Leaky ReLU activation (alpha = 0.01)
 *     reludrv()    - Leaky ReLU derivative
 *     nint_fp()    - Round Q8.8 fixed-point value to nearest integer
 *
 *   Network operations:
 *     initwe()     - Initialize weights with He scaling and deterministic
 *                    LCG PRNG (seed 887 for W1, seed 883 for W2)
 *     forwrd()     - Forward pass for one sample through the 2-4-1 network
 *     update()     - Apply accumulated gradients to all weights and biases
 *     zerogr()     - Zero all gradient arrays
 *
 * REFERENCES
 * ----------
 *   ataritron.c - Training loop entry point
 *   mlp.h       - Header with constants, types, and macro aliases
 */
#include <stdio.h>
#include "mlp.h"

/* ---- Global COMMON (from Fortran) block instances ---- */
Data   data;
Wghts  wghts;
Consts consts;
Layer  layer;

/*
 * sig_table - Sigmoid activation lookup table
 *
 * CONSTRUCTION
 * -----------
 *   33 entries, indexed 0 through 32.
 *   Each entry i corresponds to input x = -8.0 + i * 0.5.
 *   Value[i] = round(sigmoid(x) * 256), stored in Q8.8 format.
 *
 *   idx   x        sigmoid(x)     * 256     rounded
 *   ---   ---      ----------     -----     -------
 *     0   -8.0     0.000335       0.086       0
 *     1   -7.5     0.000546       0.140       0
 *     2   -7.0     0.000911       0.233       0
 *     3   -6.5     0.001509       0.386       0
 *     4   -6.0     0.002479       0.635       1
 *     5   -5.5     0.004097       1.049       1
 *     6   -5.0     0.006693       1.714       2
 *     7   -4.5     0.011109       2.844       3
 *     8   -4.0     0.017986       4.604       5
 *     9   -3.5     0.029538       7.561       7
 *    10   -3.0     0.047426      12.131      12
 *    11   -2.5     0.073110      18.716      19
 *    12   -2.0     0.119203      30.526      31
 *    13   -1.5     0.182426      46.691      47
 *    14   -1.0     0.268941      68.850      69
 *    15   -0.5     0.377541      96.671      97
 *    16    0.0     0.500000     128.000     128
 *    17    0.5     0.622459     159.329     159
 *    18    1.0     0.731059     187.150     187
 *    19    1.5     0.817574     209.341     209
 *    20    2.0     0.880797     225.526     225
 *    21    2.5     0.926890     237.341     237
 *    22    3.0     0.952574     243.918     244
 *    23    3.5     0.970462     248.438     249
 *    24    4.0     0.982014     251.456     251
 *    25    4.5     0.988891     253.156     253
 *    26    5.0     0.993307     254.341     254
 *    27    5.5     0.995903     254.918     255
 *    28    6.0     0.997521     255.364     255
 *    29    6.5     0.998491     255.614     255
 *    30    7.0     0.999089     255.767     255
 *    31    7.5     0.999454     255.860     255
 *    32    8.0     0.999665     255.914     256
 *
 * USAGE
 * ----
 *   sigmoi() uses this table for fast sigmoid evaluation on the
 *   6502 (no floating-point or exponential needed). For inputs
 *   outside [-8.0, +8.0], the value is clamped to FP_ZERO or
 *   FP_ONE. Within the table range, linear interpolation between
 *   adjacent entries provides sub-entry precision.
 */
fixed sig_table[SIG_TABLE_SIZE] = {
      0,   0,   0,   0,   1,   1,   2,   3,
      5,   7,  12,  19,  31,  47,  69,  97,
    128, 159, 187, 209, 225, 237, 244, 249,
    251, 253, 254, 255, 255, 255, 256, 256,
    256
};

/*
 * sigmoi - Evaluate the sigmoid function using a lookup table
 *
 * INPUTS:
 *   x  - Q8.8 fixed-point input value
 *
 * RETURNS:
 *   Q8.8 fixed-point sigmoid(x) in range [0, 256]
 *
 * FUNCTIONALITY:
 *   Computes sigmoid(x) = 1 / (1 + exp(-x)) without using any
 *   floating-point or exponential operations, making it suitable
 *   for the constrained 6502 environment.
 *
 *   For inputs outside the table range [-8.0, +8.0], the result
 *   is clamped to FP_ZERO (for x <= -8.0) or FP_ONE (for x >= +8.0).
 *
 *   Within the table range, the function:
 *   1. Computes the table index: idx = (x - (-2048)) / 128
 *   2. Extracts the fractional part for interpolation
 *   3. Looks up adjacent entries lo and hi from sig_table[]
 *   4. Linearly interpolates: result = lo + (hi - lo) * frac / 128
 *
 *   The lookup table contains 33 precomputed values of sigmoid(x)
 *   for x from -8.0 to +8.0 in steps of 0.5, stored in Q8.8
 *   format (multiplied by 256 and rounded).
 */
fixed sigmoi(fixed x)
{
    int idx;
    fixed lo, hi, frac;

    if (x >= SIG_CLIP)      return FP_ONE;
    if (x <= -SIG_CLIP)     return FP_ZERO;
    if (x >= 2048)          return sig_table[SIG_TABLE_SIZE - 1];
    if (x <= SIG_TABLE_MIN) return sig_table[0];

    idx  = (x - SIG_TABLE_MIN) / SIG_TABLE_STEP;
    frac = (x - SIG_TABLE_MIN) % SIG_TABLE_STEP;

    lo = sig_table[idx];
    hi = sig_table[idx + 1];
    /* interpolate: lo + (hi-lo) * frac/128, all in Q8.8 units */
    return lo + (fixed)(((int)(hi - lo) * frac) >> 7);
}

/*
 * sigdrv - Evaluate the derivative of the sigmoid function
 *
 * INPUTS:
 *   x  - Q8.8 fixed-point input value
 *
 * RETURNS:
 *   Q8.8 fixed-point sigmoid'(x) = s * (1 - s), where s = sigmoid(x)
 *   in range [0, 64] (maximum derivative is 0.25 at x = 0, stored as 64)
 *
 * FUNCTIONALITY:
 *   Computes the derivative of the sigmoid function, which is needed
 *   during backpropagation. Uses the identity:
 *
 *     d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
 *
 *   First calls sigmoi(x) to compute s = sigmoid(x), then returns
 *   s * (1 - s) using FP_MUL for fixed-point multiplication.
 *
 *   The derivative is symmetric about x = 0, peaking at 0.25 (64
 *   in Q8.8) when x = 0, and approaching 0 as |x| increases.
 */
fixed sigdrv(fixed x)
{
    fixed s = sigmoi(x);
    return FP_MUL(s, FP_ONE - s);
}

/*
 * tanhac / tanhdr - Tanh activation and derivative (stub functions)
 *
 * NOTE:
 *   These functions are NOT used in the MLP network. They are
 *   included solely to maintain traceability back to the original
 *   XORTRAN.FOR Fortran source, which defined these functions.
 *   They are stubbed to identity/constant returns to satisfy the
 *   compiler without adding unnecessary code size.
 *
 * tanhac - Tanh activation (stub)
 *
 *   INPUTS:
 *     x  - Q8.8 fixed-point input value (ignored)
 *
 *   RETURNS:
 *     x unchanged (identity function, not actual tanh)
 *
 *   FUNCTIONALITY:
 *     Stub implementation. Returns the input unchanged. The actual
 *     tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is not
 *     implemented to avoid floating-point overhead on the 6502.
 *
 * tanhdr - Tanh derivative (stub)
 *
 *   INPUTS:
 *     x  - Q8.8 fixed-point input value (ignored)
 *
 *   RETURNS:
 *     FP_ONE (1.0 in Q8.8)
 *
 *   FUNCTIONALITY:
 *     Stub implementation. Returns 1.0. The actual tanh derivative
 *     is 1 - tanh(x)^2, but this is not computed here.
 */
fixed tanhac(fixed x) { return x; }
fixed tanhdr(fixed x) { (void)x; return FP_ONE; }

/*
 * relu - Leaky ReLU activation function
 *
 * INPUTS:
 *   x  - Q8.8 fixed-point input value (pre-activation neuron output)
 *
 * RETURNS:
 *   Q8.8 fixed-point Leaky ReLU(x):
 *     x,              if x > 0
 *     ALPHA * x,      if x <= 0  (ALPHA = 3, i.e., 0.0117 in Q8.8)
 *
 * FUNCTIONALITY:
 *   Applies the Leaky ReLU activation to the input. For positive
 *   inputs, the value passes through unchanged. For zero or negative
 *   inputs, the value is scaled by ALPHA (0.01 in floating-point),
 *   allowing a small gradient to flow during backpropagation instead
 *   of being completely blocked as with standard ReLU.
 *
 *   This helps mitigate the "dying ReLU" problem where neurons can
 *   get permanently stuck at zero output.
 */
fixed relu(fixed x)
{
    if (x > FP_ZERO)
        return x;
    return FP_MUL(ALPHA, x);
}

/*
 * reludrv - Leaky ReLU derivative function
 *
 * INPUTS:
 *   x  - Q8.8 fixed-point input value (pre-activation neuron output)
 *
 * RETURNS:
 *   Q8.8 fixed-point derivative of Leaky ReLU:
 *     FP_ONE (256),     if x > 0
 *     ALPHA (3),        if x <= 0  (i.e., 0.0117 in Q8.8)
 *
 * FUNCTIONALITY:
 *   Computes the derivative of the Leaky ReLU activation function,
 *   which is needed during backpropagation. The derivative is:
 *
 *     1.0,  if x > 0   (positive inputs pass gradient unchanged)
 *     0.01, if x <= 0  (small slope for negative inputs)
 *
 *   Unlike standard ReLU derivative (which returns 0 for x <= 0),
 *   Leaky ReLU returns ALPHA, allowing a small gradient to flow
 *   even for negative inputs. This prevents neurons from dying
 *   during training when their pre-activation values are negative.
 */
fixed reludrv(fixed x)
{
    if (x > FP_ZERO)
        return FP_ONE;
    return ALPHA;
}

/*
 * nint_fp - Round a Q8.8 fixed-point value to the nearest integer
 *
 * INPUTS:
 *   x  - Q8.8 fixed-point value to round
 *
 * RETURNS:
 *   int - nearest integer value (rounded to nearest, ties round away from zero)
 *
 * FUNCTIONALITY:
 *   Converts a Q8.8 fixed-point number to the nearest integer by
 *   adding FP_HALF (0.5) to the value and then shifting right by
 *   FRAC_BITS (8) to discard the fractional part.
 *
 *   For positive values: (x + 128) >> 8
 *   For negative values: -( (-x + 128) >> 8 )
 *
 *   The sign-aware approach ensures correct rounding for negative
 *   numbers. For example, -1.6 (Q8.8: -409) rounds to -2, and
 *   -1.4 (Q8.8: -358) rounds to -1.
 *
 *   This function is used to convert network outputs and inputs
 *   to integer form for display and comparison with expected values.
 */
int nint_fp(fixed x)
{
    if (x >= 0)
        return (int)((x + FP_HALF) >> FRAC_BITS);
    else
        return -(int)((-x + FP_HALF) >> FRAC_BITS);
}

/*
 * initwe - Initialize network weights and biases
 *
 * INPUTS:
 *   None
 *
 * RETURNS:
 *   None (modifies global weight arrays W1, B1, W2, B2 in place)
 *
 * FUNCTIONALITY:
 *   Initializes all network weights and biases using He (Kaiming)
 *   scaling with a deterministic linear congruential generator (LCG)
 *   pseudo-random number generator. This ensures reproducible results
 *   across runs and platforms.
 *
 *   Weight initialization (W1: input -> hidden, 2x4 matrix):
 *     - LCG seed starts at 887, increments by 3277 each step
 *     - Random value: rnd = ((seed * 7 + 1) mod 32767) / 32767
 *     - Weight: W1[i][j] = (2 * (rnd - 0.5)) * HESCAL1
 *     - HESCAL1 = sqrt(2 / (fan_in + fan_out)) = sqrt(2 / 6) = 0.5774
 *     - Biases B1 initialized to zero
 *
 *   Weight initialization (W2: hidden -> output, 4x1 matrix):
 *     - LCG seed starts at 883, increments by 3277 each step
 *     - Random value: rnd = ((seed * 7 + 1) mod 32767) / 32767
 *     - Weight: W2[i][0] = (2 * (rnd - 0.5)) * HESCAL2
 *     - HESCAL2 = sqrt(2 / (fan_in + fan_out)) = sqrt(2 / 5) = 0.6325
 *     - Bias B2 initialized to zero
 *
 *   The He scaling factor adjusts the weight standard deviation based
 *   on the number of input connections, helping maintain gradient
 *   variance during training and preventing vanishing/exploding
 *   gradients in deeper networks.
 *
 *   The deterministic PRNG (seed 887 for W1, 883 for W2) ensures
 *   that the same weights are produced every time the program runs,
 *   which is important for reproducibility and debugging.
 */
void initwe(void)
{
    long seed;
    long raw;
    fixed rnd;
    int i, j;

    /* Initialize W1: seed 887, step 3277 */
    seed = 887L;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            raw = (seed * 7L + 1L) % 32767L;
            rnd = (fixed)((raw * (long)FP_ONE) / 32767L);
            W1[i][j] = FP_MUL(2 * (rnd - FP_HALF), HESCAL1);
            seed += 3277L;
        }
    }

    /* Initialize B1 to zero */
    for (i = 0; i < 4; i++) {
        B1[i] = FP_ZERO;
    }

    /* Initialize W2: seed 883, step 3277 */
    seed = 883L;
    for (i = 0; i < 4; i++) {
        raw = (seed * 7L + 1L) % 32767L;
        rnd = (fixed)((raw * (long)FP_ONE) / 32767L);
        W2[i][0] = FP_MUL(2 * (rnd - FP_HALF), HESCAL2);
        seed += 3277L;
    }

    /* Initialize B2 to zero */
    B2[0] = FP_ZERO;
}

/*
 * forwrd - Forward pass through the 2-4-1 MLP for one training sample
 *
 * INPUTS:
 *   isample  - 1-based index of the training sample to process
 *              (1 = 0/0, 2 = 0/1, 3 = 1/0, 4 = 1/1)
 *
 * RETURNS:
 *   None (modifies global layer arrays Z1, A1, Z2, A2 in place)
 *
 * FUNCTIONALITY:
 *   Computes the forward pass of the neural network for a single
 *   input sample. The network has two layers:
 *
 *   Layer 1 (hidden, 4 neurons):
 *     For each neuron i (0 to 3):
 *       1. Compute weighted sum: Z1[i] = sum(INPUTS[isample-1][j] * W1[j][i])
 *          for j = 0 to 1 (two input features)
 *       2. Add bias: Z1[i] += B1[i]
 *       3. Apply Leaky ReLU activation: A1[i] = relu(Z1[i])
 *
 *   Layer 2 (output, 1 neuron):
 *     1. Compute weighted sum: Z2[0] = sum(A1[j] * W2[j][0])
 *        for j = 0 to 3 (four hidden activations)
 *     2. Add bias: Z2[0] += B2[0]
 *     3. Apply Leaky ReLU activation: A2[0] = relu(Z2[0])
 *
 *   The pre-activation values (Z1, Z2) and post-activation values
 *   (A1, A2) are stored in global COMMON block arrays and are used
 *   by the backpropagation routine to compute gradients.
 *
 *   Note: isample is 1-based (matching Fortran convention), so the
 *   INPUTS array is accessed as INPUTS[isample-1].
 */
void forwrd(int isample)
{
    int i, j;

    /* Layer 1: hidden */
    for (i = 0; i < 4; i++) {
        Z1[i] = FP_ZERO;
        for (j = 0; j < 2; j++) {
            Z1[i] += FP_MUL(INPUTS[isample - 1][j], W1[j][i]);
        }
        Z1[i] += B1[i];
        A1[i] = relu(Z1[i]);
    }

    /* Layer 2: output */
    Z2[0] = FP_ZERO;
    for (j = 0; j < 4; j++) {
        Z2[0] += FP_MUL(A1[j], W2[j][0]);
    }
    Z2[0] += B2[0];
    A2[0] = relu(Z2[0]);
}

/*
 * update - Apply accumulated gradients to network weights and biases
 *
 * INPUTS:
 *   dw1, dw2 - Gradient arrays for weights W1 and W2
 *   db1, db2 - Gradient arrays for biases B1 and B2
 *   lr       - Learning rate
 *
 * RETURNS:
 *   None (modifies global weight and bias arrays in place)
 *
 * FUNCTIONALITY:
 *   Updates the network weights and biases using the provided gradients
 *   and learning rate. The update rule is:
 *     W <- W + lr * dW
 *     B <- B + lr * dB
 */
void update(fixed *dw1, fixed *dw2, fixed *db1, fixed *db2, fixed lr)
{
    int i, j;

    for (i = 0; i < 4; i++) {
        W2[i][0] += FP_MUL(lr, dw2[i]);
    }
    B2[0] += FP_MUL(lr, db2[0]);

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            W1[i][j] += FP_MUL(lr, dw1[i * 4 + j]);
        }
    }

    for (i = 0; i < 4; i++) {
        B1[i] += FP_MUL(lr, db1[i]);
    }
}

/*
 * zerogr - Zero out gradient arrays for weights and biases
 *
 * INPUTS:
 *   dw1, dw2 - Gradient arrays for weights W1 and W2
 *   db1, db2 - Gradient arrays for biases B1 and B2
 *
 * RETURNS:
 *   None (modifies gradient arrays in place)
 *
 * FUNCTIONALITY:
 *   Sets all elements of the provided gradient arrays to zero.
 */
void zerogr(fixed *dw1, fixed *dw2, fixed *db1, fixed *db2)
{
    int i, j;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            dw1[i * 4 + j] = FP_ZERO;
        }
    }
    for (i = 0; i < 4; i++) {
        dw2[i] = FP_ZERO;
    }
    for (i = 0; i < 4; i++) {
        db1[i] = FP_ZERO;
    }
    db2[0] = FP_ZERO;
}