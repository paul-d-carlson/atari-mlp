/*
 * mlp.h - Multi-Layer Perceptron Header for CC65 / Atari 8-bit
 * 
 * PURPOSE
 * -------
 * This header file defines the data structures, constants, and function
 * declarations for the Multi-Layer Perceptron: It defined a 2-4-1 MLP
 * trained to learn the XOR boolean function. It is a C port of the
 * original XORTRAN.FOR Fortran program, adapted for the CC65 cross-
 * compiler targeting the Atari 8-bit family (6502 processor).
 *
 * Q8.8 FIXED-POINT FORMAT
 * ----------------------
 *   All arithmetic uses Q8.8 fixed-point: 16-bit signed integers
 *   representing values multiplied by 256. Range: -128.0 to +127.996.
 *   Precision: 1/256 ~ 0.0039. FP_MUL uses a 32-bit long intermediate
 *   to prevent overflow during multiplication.
 *
 *   FP_ONE   = 256   (1.0 in Q8.8)
 *   FP_HALF  = 128   (0.5 in Q8.8)
 *   FP_ZERO  = 0
 *
 * NETWORK CONSTANTS
 * ----------------
 *   EPOCHS   = 100   (total training epochs)
 *   SAMPLES  = 4     (XOR input pairs)
 *   LR_INIT  = 128   (0.5 initial learning rate)
 *   LR_DECAY = 218   (0.85 decay factor, applied every LR_STEP epochs)
 *   LR_STEP  = 20    (decay interval)
 *   ALPHA    = 3     (0.01 leaky ReLU slope)
 *   SIG_CLIP = 5120  (20.0 sigmoid clamping threshold)
 *   HESCAL1  = 148   (He scaling for W1: sqrt(2/6))
 *   HESCAL2  = 162   (He scaling for W2: sqrt(2/5))
 *
 * DATA STRUCTURES
 * --------------
 *   DataCommon  - COMMON /DATA/ (INPUTS, TARGETS)
 *   WghtsCommon - COMMON /WGHTS/ (W1, B1, W2, B2)
 *   ConstsCommon- COMMON /CONSTS/ (EXPLIM, LR)
 *   LayerCommon - COMMON /LAYER/ (Z1, A1, Z2, A2)
 *
 *   Macros map struct members to Fortran-style names (INPUTS, W1, etc.)
 *   for direct correspondence with the original XORTRAN.FOR source.
 *
 * REFERENCES
 * ----------
 *   ataritron.c - Training loop entry point
 *   mlp.c       - MLP core implementation
 *   XORTRAN.FOR - Original Fortran source
 */
#ifndef MLP_H
#define MLP_H

/*
 * Fixed-point Q8.8 arithmetic
 *   typedef int fixed  (16-bit signed on CC65 / 6502)
 *   Range:     -128.0 to +127.996
 *   Precision: 1/256  ~ 0.0039
 *   FP_MUL uses a 32-bit long intermediate to prevent overflow.
 */
typedef int fixed;

#define FRAC_BITS   8
#define FP_ONE      256         /* 1.0 in Q8.8 */
#define FP_HALF     128         /* 0.5 in Q8.8 */
#define FP_ZERO     0

/* Multiply two Q8.8 values, result Q8.8 (32-bit intermediate) */
#define FP_MUL(a, b)    ((fixed)(((long)(a) * (long)(b)) >> FRAC_BITS))

/* Divide two Q8.8 values, result Q8.8 */
#define FP_DIV(a, b)    ((fixed)(((long)(a) << FRAC_BITS) / (long)(b)))

/* Convert an integer n to Q8.8 */
#define INT_TO_FP(n)    ((fixed)((n) << FRAC_BITS))

/* ---- Network constants ---- */
#define EPOCHS      100
#define SAMPLES     4
#define EXPLIM      10
#define LR_STEP     20

/*
 * Fixed-point constants (Q8.8, precomputed as integers):
 *   LR_INIT  = 0.5    * 256 = 128   (exact)
 *   LR_DECAY = 0.85   * 256 = 217.6 -> 218
 *   ALPHA    = 0.01   * 256 = 2.56  -> 3   (leaky ReLU slope)
 *   SIG_CLIP = 20.0   * 256 = 5120
 *   HESCAL1  = sqrt(2/(2+4)) * 256 = 0.5774 * 256 = 147.8 -> 148
 *   HESCAL2  = sqrt(2/(4+1)) * 256 = 0.6325 * 256 = 161.9 -> 162
 */
#define LR_INIT     128
#define LR_DECAY    218
#define ALPHA         3
#define SIG_CLIP   5120
#define HESCAL1     148
#define HESCAL2     162

/*
 * Sigmoid lookup table: 33 entries, x from -8.0 to +8.0, step 0.5.
 * Values stored in Q8.8.  Outside this range sigmoid saturates to 0/FP_ONE.
 */
#define SIG_TABLE_SIZE  33
#define SIG_TABLE_MIN   (-2048)     /* -8.0 * 256 */
#define SIG_TABLE_STEP  128         /*  0.5 * 256 */

extern fixed sig_table[SIG_TABLE_SIZE];

/* ---- COMMON /DATA/ ---- */
typedef struct {
    fixed f_INPUTS[4][2];   /* [sample][feature] */
    fixed f_TARGETS[4][1];  /* [sample][0]        */
} Data;

/* ---- COMMON /WGHTS/ ---- */
typedef struct {
    fixed f_W1[2][4];       /* input -> hidden weights  */
    fixed f_B1[4];          /* hidden biases            */
    fixed f_W2[4][1];       /* hidden -> output weights */
    fixed f_B2[1];          /* output bias              */
} Wghts;

/* ---- COMMON /CONSTS/ ---- */
typedef struct {
    int   c_explim;
    fixed c_lr;
} Consts;

/* ---- COMMON /LAYER/ ---- */
typedef struct {
    fixed f_Z1[4];          /* hidden pre-activation */
    fixed f_A1[4];          /* hidden activation     */
    fixed f_Z2[1];          /* output pre-activation */
    fixed f_A2[1];          /* output activation     */
} Layer;

/* ---- Global common block instances ---- */
extern Data   data;
extern Wghts  wghts;
extern Consts consts;
extern Layer  layer;

/* ---- Macros to match Fortran naming ---- */
#define INPUTS   data.f_INPUTS
#define TARGETS  data.f_TARGETS

#define W1       wghts.f_W1
#define B1       wghts.f_B1
#define W2       wghts.f_W2
#define B2       wghts.f_B2

#define LR       consts.c_lr

#define Z1       layer.f_Z1
#define A1       layer.f_A1
#define Z2       layer.f_Z2
#define A2       layer.f_A2

/* ---- Function declarations ---- */
void  initwe(void);
void  forwrd(int isample);
void  update(fixed *dw1, fixed *dw2, fixed *db1, fixed *db2, fixed lr);
void  zerogr(fixed *dw1, fixed *dw2, fixed *db1, fixed *db2);

fixed sigmoi(fixed x);
fixed sigdrv(fixed x);
fixed relu(fixed x);
fixed reludrv(fixed x);
int   nint_fp(fixed x);

#endif /* MLP_H */