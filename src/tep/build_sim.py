"""
Build a parameterised Tennessee Eastman simulator from the reference
Fortran and expose it to Python through f2py.

Why not the packaged d00..d21 data sets: their fault magnitudes and noise
levels are fixed, so the bound-tightness / SNR sweep (E5) cannot be run on
them.  This build adds three knobs the original code does not have:

  FMAG(20)   per-fault magnitude multiplier for the seven step faults
             (1.0 reproduces the benchmark magnitude exactly)
  XNSG       measurement-noise gain
  SSPANG     random-walk amplitude gain, for the random-variation faults

plus an explicit RNG seed, so runs are reproducible and can be repeated
across seeds.

Sources (downloaded, unmodified copies kept in data/raw/tep_src):
  teprob.f      Downs & Vogel plant model
  temain_mod.f  Russell / Chiang / Braatz closed-loop control scheme
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "tep_src")
BUILD = os.path.join(ROOT, "src", "tep", "fortran")

# ---- the seven step-magnitude literals that become FMAG-scalable --------
STEP_PATCHES = [
    ("      XST(1,4)=TESUB8(1,TIME)-IDV(1)*0.03D0\n     .-IDV(2)*2.43719D-3\n",
     "      XST(1,4)=TESUB8(1,TIME)-IDV(1)*0.03D0*FMAG(1)\n"
     "     .-IDV(2)*2.43719D-3*FMAG(2)\n"),
    ("      XST(2,4)=TESUB8(2,TIME)+IDV(2)*0.005D0\n",
     "      XST(2,4)=TESUB8(2,TIME)+IDV(2)*0.005D0*FMAG(2)\n"),
    ("      TST(1)=TESUB8(3,TIME)+IDV(3)*5.D0\n",
     "      TST(1)=TESUB8(3,TIME)+IDV(3)*5.D0*FMAG(3)\n"),
    ("      TCWR=TESUB8(5,TIME)+IDV(4)*5.D0\n",
     "      TCWR=TESUB8(5,TIME)+IDV(4)*5.D0*FMAG(4)\n"),
    ("      TCWS=TESUB8(6,TIME)+IDV(5)*5.D0\n",
     "      TCWS=TESUB8(6,TIME)+IDV(5)*5.D0*FMAG(5)\n"),
    ("      FTM(3)=VPOS(3)*(1.D0-IDV(6))*VRNG(3)/100.0\n",
     "      FTM(3)=VPOS(3)*(1.D0-IDV(6)*FMAG(6))*VRNG(3)/100.0\n"),
    ("      FTM(4)=VPOS(4)*(1.D0-IDV(7)*0.2D0)\n",
     "      FTM(4)=VPOS(4)*(1.D0-IDV(7)*0.2D0*FMAG(7))\n"),
]

# measurement noise: scale the standard deviation at the three call sites
NOISE_PATCH = ("      CALL TESUB6(XNS(I),XMNS)\n",
               "      CALL TESUB6(XNS(I)*XNSG,XMNS)\n")

# declaration injected into TEFUNC
TEFUNC_DECL = """      DOUBLE PRECISION FMAG, XNSG
      COMMON/TEPMAG/ FMAG(20), XNSG
"""


def patch_teprob():
    src = open(os.path.join(RAW, "teprob.f"), encoding="latin-1").read()
    n = 0
    for old, new in STEP_PATCHES:
        if old not in src:
            raise SystemExit("teprob.f patch target not found:\n%r" % old)
        src = src.replace(old, new, 1)
        n += 1
    cnt = src.count(NOISE_PATCH[0])
    if cnt != 3:
        raise SystemExit("expected 3 noise call sites, found %d" % cnt)
    src = src.replace(*NOISE_PATCH)
    n += cnt

    # declare the new common block inside TEFUNC, right after its header
    marker = "      SUBROUTINE TEFUNC(NN,TIME,YY,YP)\n"
    if marker not in src:
        raise SystemExit("TEFUNC header not found")
    src = src.replace(marker, marker + TEFUNC_DECL, 1)
    n += 1
    out = os.path.join(BUILD, "teprob_m.f")
    open(out, "w", encoding="latin-1").write(src)
    print("teprob_m.f written (%d patches)" % n)


def strip_main():
    """Keep only the subroutines of temain_mod.f, dropping its PROGRAM."""
    lines = open(os.path.join(RAW, "temain_mod.f"),
                 encoding="latin-1").read().splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("SUBROUTINE CONTRL1"):
            start = i
            break
    if start is None:
        raise SystemExit("CONTRL1 not found in temain_mod.f")
    body = "".join(lines[start:])
    # the OUTPUT subroutine only writes files; keep it out of the build
    i0 = body.find("      SUBROUTINE OUTPUT")
    i1 = body.find("      SUBROUTINE INTGTR")
    if i0 < 0 or i1 < 0 or i1 < i0:
        raise SystemExit("could not locate OUTPUT/INTGTR boundary")
    body = body[:i0] + body[i1:]
    out = os.path.join(BUILD, "tectrl.f")
    open(out, "w", encoding="latin-1").write(body)
    print("tectrl.f written (%d lines, OUTPUT removed)"
          % body.count("\n"))


WRAPPER = r"""
      SUBROUTINE TESIM(NPTS,NSAMP,IDVIN,TIDV,FMAGIN,XNSGIN,SSPANG,
     .                 VSTG,SEED,NS,YM,YV,ISHUT)
C
C  Closed-loop Tennessee Eastman run with parameterised fault magnitude,
C  measurement noise and random-walk amplitude.
C
C  NPTS   number of 1-second integration steps
C  NSAMP  store one sample every NSAMP steps (180 = the 3 min benchmark rate)
C  IDVIN  fault flags, switched on at step TIDV
C  TIDV   step index at which the faults are switched on
C  FMAGIN per-fault magnitude multiplier (1.0 = benchmark magnitude)
C  XNSGIN measurement noise gain (1.0 = benchmark noise)
C  SSPANG random-walk amplitude gain (1.0 = benchmark amplitude)
C  VSTG   valve stiction deadband gain (1.0 = benchmark deadband).  The
C         stiction disturbances IDV(14), (15) and (19) are boolean flags in
C         the reference source with a fixed deadband VST(j), so without this
C         knob they have no magnitude to sweep.
C  SEED   RNG seed; <=0 keeps the built-in default
C  NS     number of samples to store (= NPTS/NSAMP)
C  YM     out, (NS,41) measurements
C  YV     out, (NS,12) manipulated variables
C  ISHUT  out, step index of a plant shutdown, 0 if none
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER NPTS,NSAMP,TIDV,NS,ISHUT,I,J,K,NN
      INTEGER IDVIN(20)
      DOUBLE PRECISION FMAGIN(20),XNSGIN,SSPANG,VSTG,SEED
      DOUBLE PRECISION YM(NS,41),YV(NS,12)
Cf2py intent(in) npts,nsamp,idvin,tidv,fmagin,xnsgin,sspang,vstg,seed,ns
Cf2py intent(out) ym,yv,ishut
      DOUBLE PRECISION YY(50),YP(50)
      DOUBLE PRECISION XMEAS,XMV
      COMMON/PV/ XMEAS(41),XMV(12)
      INTEGER IDV
      COMMON/DVEC/ IDV(20)
      DOUBLE PRECISION SETPT,DELTAT
      COMMON/CTRLALL/ SETPT(20),DELTAT
      INTEGER FLAG
      COMMON/FLAG6/ FLAG
      COMMON/CTRL1/ GAIN1,ERROLD1
      COMMON/CTRL2/ GAIN2,ERROLD2
      COMMON/CTRL3/ GAIN3,ERROLD3
      COMMON/CTRL4/ GAIN4,ERROLD4
      COMMON/CTRL5/ GAIN5,TAUI5,ERROLD5
      COMMON/CTRL6/ GAIN6,ERROLD6
      COMMON/CTRL7/ GAIN7,ERROLD7
      COMMON/CTRL8/ GAIN8,ERROLD8
      COMMON/CTRL9/ GAIN9,ERROLD9
      COMMON/CTRL10/ GAIN10,TAUI10,ERROLD10
      COMMON/CTRL11/ GAIN11,TAUI11,ERROLD11
      COMMON/CTRL13/ GAIN13,TAUI13,ERROLD13
      COMMON/CTRL14/ GAIN14,TAUI14,ERROLD14
      COMMON/CTRL15/ GAIN15,TAUI15,ERROLD15
      COMMON/CTRL16/ GAIN16,TAUI16,ERROLD16
      COMMON/CTRL17/ GAIN17,TAUI17,ERROLD17
      COMMON/CTRL18/ GAIN18,TAUI18,ERROLD18
      COMMON/CTRL19/ GAIN19,TAUI19,ERROLD19
      COMMON/CTRL20/ GAIN20,TAUI20,ERROLD20
      COMMON/CTRL22/ GAIN22,TAUI22,ERROLD22
      DOUBLE PRECISION G
      COMMON/RANDSD/ G
      INTEGER IDVWLK
      COMMON/WLK/ ADIST(12),BDIST(12),CDIST(12),DDIST(12),
     .TLAST(12),TNEXT(12),HSPAN(12),HZERO(12),SSPAN(12),
     .SZERO(12),SPSPAN(12),IDVWLK(12)
      DOUBLE PRECISION FMAG,XNSG
      COMMON/TEPMAG/ FMAG(20),XNSG
      DOUBLE PRECISION UCLR,UCVR,UTLR,UTVR,XLR,XVR,ETR,ESR,TCR,TKR,DLR,
     .VLR,VVR,VTR,PTR,PPR,CRXR,RR,RH,FWR,TWR,QUR,HWR,UAR,UCLS,UCVS,UTLS,
     .UTVS,XLS,XVS,ETS,ESS,TCS,TKS,DLS,VLS,VVS,VTS,PTS,PPS,FWS,TWS,QUS,
     .HWS,UCLC,UTLC,XLC,ETC,ESC,TCC,DLC,VLC,VTC,QUC,UCVV,UTVV,XVV,ETV,
     .ESV,TCV,TKV,VTV,PTV,VCV,VRNG,VTAU,FTM,FCM,XST,XMWS,HST,TST,SFR,
     .CPFLMX,CPPRMX,CPDH,TCWR,TCWS,HTR,AGSP,XDEL,XNS,TGAS,TPROD,VST
      INTEGER IVST
      COMMON/TEPROC/
     .UCLR(8),UCVR(8),UTLR,UTVR,XLR(8),XVR(8),ETR,ESR,TCR,TKR,DLR,VLR,
     .VVR,VTR,PTR,PPR(8),CRXR(8),RR(4),RH,FWR,TWR,QUR,HWR,UAR,
     .UCLS(8),UCVS(8),UTLS,UTVS,XLS(8),XVS(8),ETS,ESS,TCS,TKS,DLS,VLS,
     .VVS,VTS,PTS,PPS(8),FWS,TWS,QUS,HWS,
     .UCLC(8),UTLC,XLC(8),ETC,ESC,TCC,DLC,VLC,VTC,QUC,
     .UCVV(8),UTVV,XVV(8),ETV,ESV,TCV,TKV,VTV,PTV,VCV(12),VRNG(12),
     .VTAU(12),FTM(13),FCM(8,13),XST(8,13),XMWS(13),HST(13),TST(13),
     .SFR(8),CPFLMX,CPPRMX,CPDH,TCWR,TCWS,HTR(3),AGSP,XDEL(41),XNS(41),
     .TGAS,TPROD,VST(12),IVST(12)
C
      NN=50
      DO 5 I=1,20
         FMAG(I)=FMAGIN(I)
    5 CONTINUE
      XNSG=XNSGIN
      DELTAT=1.D0/3600.D0
      CALL TEINIT(NN,TIME,YY,YP)
      IF(SEED.GT.0.D0) G=SEED
      DO 7 I=1,12
         SSPAN(I)=SSPAN(I)*SSPANG
         SPSPAN(I)=SPSPAN(I)*SSPANG
         VST(I)=VST(I)*VSTG
    7 CONTINUE
C
      SETPT(1)=3664.0
      GAIN1=1.0
      ERROLD1=0.0
      SETPT(2)=4509.3
      GAIN2=1.0
      ERROLD2=0.0
      SETPT(3)=.25052
      GAIN3=1.
      ERROLD3=0.0
      SETPT(4)=9.3477
      GAIN4=1.
      ERROLD4=0.0
      SETPT(5)=26.902
      GAIN5=-0.083
      TAUI5=1./3600.
      ERROLD5=0.0
      SETPT(6)=0.33712
      GAIN6=1.22
      ERROLD6=0.0
      SETPT(7)=50.0
      GAIN7=-2.06
      ERROLD7=0.0
      SETPT(8)=50.0
      GAIN8=-1.62
      ERROLD8=0.0
      SETPT(9)=230.31
      GAIN9=0.41
      ERROLD9=0.0
      SETPT(10)=94.599
      GAIN10=-0.156*10.
      TAUI10=1452./3600.
      ERROLD10=0.0
      SETPT(11)=22.949
      GAIN11=1.09
      TAUI11=2600./3600.
      ERROLD11=0.0
      SETPT(13)=32.188
      GAIN13=18.
      TAUI13=3168./3600.
      ERROLD13=0.0
      SETPT(14)=6.8820
      GAIN14=8.3
      TAUI14=3168.0/3600.
      ERROLD14=0.0
      SETPT(15)=18.776
      GAIN15=2.37
      TAUI15=5069./3600.
      ERROLD15=0.0
      SETPT(16)=65.731
      GAIN16=1.69/10.
      TAUI16=236./3600.
      ERROLD16=0.0
      SETPT(17)=75.000
      GAIN17=11.1/10.
      TAUI17=3168./3600.
      ERROLD17=0.0
      SETPT(18)=120.40
      GAIN18=2.83*10.
      TAUI18=982./3600.
      ERROLD18=0.0
      SETPT(19)=13.823
      GAIN19=-83.2/5./3.
      TAUI19=6336./3600.
      ERROLD19=0.0
      SETPT(20)=0.83570
      GAIN20=-16.3/5.
      TAUI20=12408./3600.
      ERROLD20=0.0
      SETPT(12)=2633.7
      GAIN22=-1.0*5.
      TAUI22=1000./3600.
      ERROLD22=0.0
      FLAG=0
C
      XMV(1)=63.053
      XMV(2)=53.980
      XMV(3)=24.644
      XMV(4)=61.302
      XMV(5)=22.210
      XMV(6)=40.064
      XMV(7)=38.100
      XMV(8)=46.534
      XMV(9)=47.446
      XMV(10)=41.106
      XMV(11)=18.114
      DO 10 I=1,20
         IDV(I)=0
   10 CONTINUE
      ISHUT=0
      K=0
C
      DO 1000 I=1,NPTS
         IF(I.GE.TIDV)THEN
            DO 20 J=1,20
               IDV(J)=IDVIN(J)
   20       CONTINUE
         ENDIF
         IF(MOD(I,3).EQ.0)THEN
            CALL CONTRL1
            CALL CONTRL2
            CALL CONTRL3
            CALL CONTRL4
            CALL CONTRL5
            CALL CONTRL6
            CALL CONTRL7
            CALL CONTRL8
            CALL CONTRL9
            CALL CONTRL10
            CALL CONTRL11
            CALL CONTRL16
            CALL CONTRL17
            CALL CONTRL18
         ENDIF
         IF(MOD(I,360).EQ.0)THEN
            CALL CONTRL13
            CALL CONTRL14
            CALL CONTRL15
            CALL CONTRL19
         ENDIF
         IF(MOD(I,900).EQ.0) CALL CONTRL20
         IF(MOD(I,NSAMP).EQ.0 .AND. K.LT.NS)THEN
            K=K+1
            DO 30 J=1,41
               YM(K,J)=XMEAS(J)
   30       CONTINUE
            DO 40 J=1,12
               YV(K,J)=XMV(J)
   40       CONTINUE
         ENDIF
         CALL INTGTR(NN,TIME,DELTAT,YY,YP)
         CALL CONSHAND
         IF(XMEAS(7).GT.3000.0 .AND. ISHUT.EQ.0) ISHUT=I
 1000 CONTINUE
      RETURN
      END
"""


def write_wrapper():
    out = os.path.join(BUILD, "tesim.f")
    open(out, "w", encoding="latin-1").write(WRAPPER.lstrip("\n"))
    print("tesim.f written")


def compile_ext():
    cmd = [sys.executable, "-m", "numpy.f2py", "-c", "-m", "tepsim",
           "tesim.f", "teprob_m.f", "tectrl.f",
           "--backend", "meson",
           "--f77flags=-ffixed-line-length-none -std=legacy -O2"]
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).splitlines()
    print("\n".join(tail[-30:]))
    if r.returncode != 0:
        raise SystemExit("f2py build failed (rc=%d)" % r.returncode)
    for f in os.listdir(BUILD):
        if f.startswith("tepsim") and f.endswith((".pyd", ".so")):
            shutil.copy(os.path.join(BUILD, f), os.path.join(ROOT, "src", "tep", f))
            print("installed", f)


if __name__ == "__main__":
    os.makedirs(BUILD, exist_ok=True)
    patch_teprob()
    strip_main()
    write_wrapper()
    compile_ext()
