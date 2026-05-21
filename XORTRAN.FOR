C  XOR MULTILAYER PERCEPTION
C  2025, DAMIEN BOUREILLE
C  CODE RELEASED UNDER THE MIT LICENSE
C  ------------------------------------------------------
C  MAIN PROGRAM
C     COMMON BLOCKS AND TYPE DECLARATIONS
      REAL INPUTS(4,2), TARGETS(4,1)
      REAL W1(2,4), B1(4), W2(4,1), B2(1)
      REAL Z1(4), A1(4), Z2(1), A2(1)
      REAL DELTA1(4), DELTA2(1)
      REAL DW1(2,4), DW2(4,1), DB1(4), DB2(1)
      REAL LOSS, ERROR, LR, DW1UPDT
      INTEGER EPOCHS, SAMPLES, I, J, EPOCH, EXPLIM
C
      COMMON /DATA/ INPUTS, TARGETS
      COMMON /WGHTS/ W1, B1, W2, B2
      COMMON /CONSTS/ EXPLIM, LR
      COMMON /LAYER/ Z1, A1, Z2, A2
C
C     CONSTANTS
      EPOCHS = 100
      SAMPLES = 4
      EXPLIM = 10
      LR = 0.5
C
C     DATA
      INPUTS(1,1)=0.0
      INPUTS(1,2)=0.0
      INPUTS(2,1)=0.0
      INPUTS(2,2)=1.0
      INPUTS(3,1)=1.0
      INPUTS(3,2)=0.0
      INPUTS(4,1)=1.0
      INPUTS(4,2)=1.0
      TARGETS(1,1)=0.0
      TARGETS(2,1)=1.0
      TARGETS(3,1)=1.0
      TARGETS(4,1)=0.0
C
C     FORMATS
9000  FORMAT(' EPOCH   LOSS(MSE)      LR')
9001  FORMAT(I5, ' ', F14.12, ' ', F8.6)
9002  FORMAT(' ')
9003  FORMAT (I2, I2, ' GOT:', F9.6, ' EXPECTED:', F2.0)
C
C     INITIALIZE WEIGHTS AND BIASES
      CALL INITWE
C
C     TRAINING LOOP
      WRITE(7, 9000)
      DO 100 EPOCH = 1, EPOCHS
          LOSS = 0.0
          CALL ZEROGR(DW1, DW2, DB1, DB2)
          DO 200 ISAMPLE = 1, SAMPLES
              CALL FORWRD(ISAMPLE)
              ERROR = TARGETS(ISAMPLE,1) - A2(1)
 70           LOSS = LOSS + ERROR * ERROR
              DELTA2(1) = ERROR * SIGDRV(Z2(1))
              DO 300 I = 1, 4
                  DW2(I,1) = DW2(I,1) + A1(I) * DELTA2(1)
 300          CONTINUE
              DB2(1) = DB2(1) + DELTA2(1)
              DO 400 I = 1, 4
                  DELTA1(I) = DELTA2(1) * W2(I,1) * RELUDRV(Z1(I))
                  DO 500 J = 1, 2
                      DW1UPDT = INPUTS(ISAMPLE,J) * DELTA1(I)       
                      DW1(J,I) = DW1(J,I) + DW1UPDT
 500              CONTINUE
                  DB1(I) = DB1(I) + DELTA1(I)
 400          CONTINUE
 200      CONTINUE
          DO 10 I = 1, 2
            DO 20 J = 1, 4
                DW1(I,J) = DW1(I,J) / 4.0
   20       CONTINUE
          CALL UPDATE(DW1, DW2, DB1, DB2, LR)
   10     CONTINUE
          DO 30 I = 1, 4
              DW2(I,1) = DW2(I,1) / 4.0
              DB1(I) = DB1(I) / 4.0
   30     CONTINUE
          DB2(1) = DB2(1) / 4.0
C          IF (EPOCH .EQ. 76) GOTO 202
C          IF (EPOCH .EQ. 80) GOTO 201
C          GOTO 203
C 201      LR = 0.3
C          GOTO 203
C 202      LR = 0.01
 203      CALL UPDATE(DW1, DW2, DB1, DB2, LR)
          LOSS = LOSS / FLOAT(SAMPLES)
          IF (EPOCH .EQ. 1.0) GOTO 204
          IF (AMOD(FLOAT(EPOCH), 20.0) .NE. 0.0) GOTO 205
            LR = LR * 0.85
 204        WRITE(7,9001) EPOCH, LOSS, LR
 205      CONTINUE
 100  CONTINUE
      WRITE(7, 9002)
      DO 600 ISAMPLE = 1, SAMPLES
        CALL FORWRD(ISAMPLE)
        ITMP1 = NINT(INPUTS(ISAMPLE,1))
        ITMP2 = NINT(INPUTS(ISAMPLE,2))
C GET ABSOLUTE VALUE IN CASE OF ROUNDING ERROR NEAR 0.00
        RTMP1 = A2(1)
        RTMP2 = TARGETS(ISAMPLE,1)
        WRITE(7, 9003) ITMP1, ITMP2, RTMP1, RTMP2
 600  CONTINUE
      STOP
      END
C
C NINT
      INTEGER FUNCTION NINT(X)
      REAL X
      IF (X .GE. 0.0) GOTO 10
      NINT = -INT(0.5 - X)
      GOTO 20
10    NINT = INT(X + 0.5)
20    RETURN
      END
C
C SIGMOI
      REAL FUNCTION SIGMOI(X)
      REAL X
      IF (X .GT. 20.0) GOTO 30
      IF (X .LT. -20.0) GOTO 40
      SIGMOI = 1.0 / (1.0 + EXP(-X))
      GOTO 50
  30  SIGMOI = 1.0
      GOTO 50
  40  SIGMOI = 0.0
  50  CONTINUE
      RETURN
      END
C
C SIGDRV
      REAL FUNCTION SIGDRV(X)
      REAL X, S, SIGMOI
      S = SIGMOI(X)
      SIGDRV = S * (1.0 - S)
      RETURN
      END
C
C TANHAC
      REAL FUNCTION TANHAC(X)
      REAL X
      TANHAC = TANH(X) 
      RETURN
      END
C
C TANHDR
      REAL FUNCTION TANHDR(X)
      REAL X, T, TANH
      T = TANH(X)
      TANHDR = 1.0 - T * T
      RETURN
      END
C
C RELU
      REAL FUNCTION RELU(X)
      REAL X, ALPHA
      ALPHA = 0.01
      IF (X .GT. 0.0) GOTO 10
      RELU = ALPHA * X
      GOTO 20
10    RELU = X
20    RETURN
      END
C      
C RELUDRV
      REAL FUNCTION RELUDRV(X)
      REAL X, ALPHA
      ALPHA = 0.01
      IF (X .GT. 0.0) GOTO 10
      RELUDRV = ALPHA
      GOTO 20
10    RELUDRV = 1.0
20    RETURN
      END
C      
C INITWE
      SUBROUTINE INITWE
      REAL W1(2,4), B1(4), W2(4,1), B2(1), RND, SEED
      COMMON /WGHTS/ W1, B1, W2, B2
      INTEGER I, J
      HESCAL1 = SQRT(2.0 / FLOAT(2 + 4))
      HESCAL2 = SQRT(2.0 / FLOAT(4 + 1))

      SEED = 887.
      DO 10 I = 1, 2
          DO 20 J = 1, 4
            RND = AMOD(SEED * 7 + 1, 32767.) / 32767.
            W1(I,J) = (RND - 0.5) * 2.0 * HESCAL1
            SEED = SEED + 3277.
C            WRITE(7, 9003) (W1(I,J), SEED)
C9003        FORMAT(F20.12, F20.12)
 20       CONTINUE
 10   CONTINUE
      DO 30 I = 1, 4
C           B2(1) = RAN(1, 4000) * 0.01
           B1(I) = 0
 30   CONTINUE

      SEED = 883.
      DO 40 I = 1, 4
           RND = AMOD(SEED * 7 + 1, 32767.) / 32767.
           W2(I,1) = (RND - 0.5) * 2.0 * HESCAL2
           SEED = SEED + 3277.
 40   CONTINUE
 9000 FORMAT(A, D20.12)
C      B2(1) = RAN(1, 4000) * 0.01
      B2(1) = 0
      RETURN
      END
C
C FORWRD
      SUBROUTINE FORWRD(ISAMPLE)
      COMMON /WGHTS/ W1(2,4), B1(4), W2(4,1), B2(1)
      COMMON /LAYER/ Z1(4), A1(4), Z2(1), A2(1)
      COMMON /DATA/ INPUTS(4,2), TARGETS(4,1)
      INTEGER I, J, ISAMPLE
      REAL W1, B1, W2, B2
      REAL Z1, A1, Z2, A2, INPUTS, TARGETS
      REAL TEMP
C
C     LAYER 1
      DO 1000 I = 1, 4
          Z1(I) = 0.0
          DO 2000 J = 1, 2
              TEMP = INPUTS(ISAMPLE,J) * W1(J,I)
              Z1(I) = Z1(I) + TEMP
2000      CONTINUE
C      BIAS
          Z1(I) = Z1(I) + B1(I)
C      RELU ACTIVATION
          A1(I) = RELU(Z1(I))
1000  CONTINUE
C
C     LAYER 2
      Z2(1) = 0.0
      DO 3000 J = 1, 4
          Z2(1) = Z2(1) + A1(J) * W2(J,1)
3000  CONTINUE
C     BIAS
      Z2(1) = Z2(1) + B2(1)
C     ACTIVATION
      A2(1) = RELU(Z2(1))
      RETURN
      END
C
C UPDATE
      SUBROUTINE UPDATE(DW1, DW2, DB1, DB2, LR)
      REAL DW1(2,4), DW2(4,1), DB1(4), DB2(1), LR
      COMMON /WGHTS/ W1(2,4), B1(4), W2(4,1), B2(1)
      REAL W1, B1, W2, B2
      INTEGER I, J
      DO 10 I = 1, 4
          W2(I,1) = W2(I,1) + LR * DW2(I,1)
 10   CONTINUE
      B2(1) = B2(1) + LR * DB2(1)
      DO 20 I = 1, 2
          DO 30 J = 1, 4
              W1(I,J) = W1(I,J) + LR * DW1(I,J)
 30       CONTINUE
 20   CONTINUE
      DO 40 I = 1, 4
          B1(I) = B1(I) + LR * DB1(I)
 40   CONTINUE
      RETURN
      END
C
C ZEROGR
      SUBROUTINE ZEROGR(DW1, DW2, DB1, DB2)
      REAL DW1(2,4), DW2(4,1), DB1(4), DB2(1)
      INTEGER I, J
      DO 10 I = 1, 2
          DO 20 J = 1, 4
              DW1(I,J) = 0.0
 20       CONTINUE
 10   CONTINUE
      DO 30 I = 1, 4
          DW2(I,1) = 0.0
 30   CONTINUE
      DO 40 I = 1, 4
          DB1(I) = 0.0
 40   CONTINUE
      DB2(1) = 0.0
      RETURN
      END