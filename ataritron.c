/*
 * ataritron.c - XORTRAN Training Entry Point for CC65 / Atari 8-bit
 *
 * PURPOSE
 * -------
 * This file implements the training loop for a 2-4-1 Multi-Layer Perceptron
 * (MLP) that learns the XOR boolean function. It is a C port of the original
 * XORTRAN.FOR Fortran program, adapted for the CC65 cross-compiler targeting
 * the Atari 8-bit family (6502 processor).
 *
 * ARCHITECTURE
 * ------------
 *   Input layer:  2 neurons  (XOR input pair: 0/0, 0/1, 1/0, 1/1)
 *   Hidden layer: 4 neurons  (Leaky ReLU activation)
 *   Output layer: 1 neuron   (Leaky ReLU activation, binary classification)
 *
 * All arithmetic uses Q8.8 fixed-point format (16-bit signed, value * 256).
 * The sigmoid function is approximated via a 33-entry lookup table with
 * linear interpolation.
 *
 * TRAINING DETAILS
 * ----------------
 *   Loss:       Mean Squared Error (MSE)
 *   Optimizer:  Batch gradient descent (full-batch, all 4 samples per epoch)
 *   LR init:    0.5, decayed by *0.85 every 20 epochs
 *   Weight init: He/Kaiming scaling with deterministic LCG PRNG
 *   Gradients:  Accumulated over all 4 samples, then applied via a
 *               three-update schedule to match the Fortran's effective
 *               learning rates (W2/B1/B2 receive 9* LR; W1 receives 6* LR).
 *
 * OUTPUT
 * ------
 *   Per-epoch: epoch number, MSE loss, current learning rate
 *   Final:     input pairs, expected targets, and network outputs
 *
 * BUILD
 * -----
 *   sh build.sh
 *   produces ataritron.xex (Atari DOS executable)
 *
 * REFERENCES
 * ----------
 *   XORTRAN.FOR   - original Fortran source
 *   mlp.h / mlp.c - MLP core: activations, weight init, forward pass, update
 */
#include <stdio.h>
#include "mlp.h"

/*
 * print_fp - Print a Q8.8 fixed-point value as a decimal string
 *
 * INPUTS:
 *   v  - Q8.8 fixed-point value to print (16-bit signed, value * 256)
 *
 * RETURNS:
 *   None (void)
 *
 * FUNCTIONALITY:
 *   Prints the fixed-point value to stdout in the format "d.dd"
 *   (e.g., 1.00, 0.50, -0.75). Handles negative values by printing
 *   a leading minus sign. The integer part is extracted via right
 *   shift by FRAC_BITS; the fractional part is extracted from the
 *   low byte, scaled by 100, and shifted right by FRAC_BITS to
 *   produce a two-digit decimal.
 */
static void print_fp(fixed v)
{
    if (v < 0) {
        putchar('-');
        v = -v;
    }
    printf("%d.%02d", v >> FRAC_BITS, ((int)(v & 0xFF) * 100) >> 8);
}

/*
 * main - training program entry point
 *
 * INPUTS:
 *   None
 *
 * RETURNS:
 *   0 on success
 *
 * FUNCTIONALITY:
 *   Sets up the XOR training dataset (4 samples: 0/0->0, 0/1->1,
 *   1/0->1, 1/1->0) in Q8.8 fixed-point format, initializes the
 *   network weights via initwe(), then runs the full training loop:
 *
 *   1. For each of EPOCHS (100) epochs:
 *      a. Zero accumulated gradients
 *      b. For each of SAMPLES (4) XOR inputs:
 *         - Forward pass through the 2-4-1 network
 *         - Compute MSE error and accumulate loss
 *         - Backpropagate deltas to compute per-sample gradients
 *      c. Apply gradients via three-update schedule (matching
 *         the original Fortran's effective learning rates)
 *      d. Decay learning rate by *0.85 every LR_STEP (20) epochs
 *      e. Print epoch number, MSE loss, and current LR
 *
 *   2. After training, run a final forward pass for each sample
 *      and print input pairs, expected targets, and network outputs
 *      for verification.
 *
 *   3. Wait for user to press RETURN before exiting.
 */
int main(void)
{
    fixed loss, error;
    fixed dw1[2][4], dw2[4][1], db1[4], db2[1];
    fixed dw1updt;
    fixed delta2_0;
    fixed delta1_i;
    int epoch, isample, i, j;
    int itmp1, itmp2;

    /* COMMON /DATA/ — XOR training data in Q8.8 */
    INPUTS[0][0] = FP_ZERO; INPUTS[0][1] = FP_ZERO;
    INPUTS[1][0] = FP_ZERO; INPUTS[1][1] = FP_ONE;
    INPUTS[2][0] = FP_ONE;  INPUTS[2][1] = FP_ZERO;
    INPUTS[3][0] = FP_ONE;  INPUTS[3][1] = FP_ONE;

    TARGETS[0][0] = FP_ZERO;
    TARGETS[1][0] = FP_ONE;
    TARGETS[2][0] = FP_ONE;
    TARGETS[3][0] = FP_ZERO;

    /* COMMON /CONSTS/ */
    LR = LR_INIT;

    initwe();

    putchar(125); /* ATASCII CLR */
    printf("       XOR MULTILAYER PERCEPTRON\n");
    printf("          ATARI 8-BIT  Q8.8 FP\n");
    printf("--------------------------------------\n\n");
    printf(" EPOCH  LOSS    LR\n");

    for (epoch = 1; epoch <= EPOCHS; epoch++) {
        loss = FP_ZERO;
        zerogr(&dw1[0][0], &dw2[0][0], &db1[0], &db2[0]);

        for (isample = 1; isample <= SAMPLES; isample++) {
            forwrd(isample);

            error = TARGETS[isample - 1][0] - A2[0];
            loss += FP_MUL(error, error);

            delta2_0 = FP_MUL(error, sigdrv(Z2[0]));

            for (i = 0; i < 4; i++) {
                dw2[i][0] += FP_MUL(A1[i], delta2_0);
            }
            db2[0] += delta2_0;

            for (i = 0; i < 4; i++) {
                delta1_i = FP_MUL(FP_MUL(delta2_0, W2[i][0]), reludrv(Z1[i]));
                for (j = 0; j < 2; j++) {
                    dw1updt = FP_MUL(INPUTS[isample - 1][j], delta1_i);
                    dw1[j][i] += dw1updt;
                }
                db1[i] += delta1_i;
            }
        }

        /*
         * Three-update schedule matching the Fortran's effective learning rates:
         *   W2 / B1 / B2 : 9x standard  (Fortran: called 3× with raw, raw, /4)
         *   W1[0] + W1[1]: 6x standard, symmetric
         *
         * Call 1 — all gradients still at raw (pre-division) scale:
         */
        update(&dw1[0][0], &dw2[0][0], &db1[0], &db2[0], LR);

        /* Average DW1, then call 2 */
        for (i = 0; i < 2; i++) {
            for (j = 0; j < 4; j++) {
                dw1[i][j] /= SAMPLES;
            }
        }
        update(&dw1[0][0], &dw2[0][0], &db1[0], &db2[0], LR);

        /* Average remaining gradients, then call 3 */
        for (i = 0; i < 4; i++) {
            dw2[i][0] /= SAMPLES;
            db1[i]    /= SAMPLES;
        }
        db2[0] /= SAMPLES;
        update(&dw1[0][0], &dw2[0][0], &db1[0], &db2[0], LR);

        loss /= SAMPLES;

        if (epoch % LR_STEP == 0) {
            LR = FP_MUL(LR, LR_DECAY);
        }

        if (epoch == 1 || epoch % LR_STEP == 0) {
            printf("%6d  ", epoch);
            print_fp(loss);
            printf("    ");
            print_fp(LR);
            printf("\n");
        }
    }

    printf("\n IN  EXPECTED   GOT\n");

    for (isample = 1; isample <= SAMPLES; isample++) {
        forwrd(isample);
        itmp1 = nint_fp(INPUTS[isample - 1][0]);
        itmp2 = nint_fp(INPUTS[isample - 1][1]);
        printf(" %d%d  ", itmp1, itmp2);
        print_fp(TARGETS[isample - 1][0]);
        printf("       ");
        print_fp(A2[0]);
        printf("\n");
    }

    printf("\nPress RETURN to exit...");
    getchar();

    return 0;
}