
Fatbin elf code:
================
arch = sm_52
code version = [1,7]
host = linux
compile_size = 64bit

	code for sm_52

Fatbin elf code:
================
arch = sm_52
code version = [1,7]
host = linux
compile_size = 64bit

	code for sm_52
		Function : _Z9vectorAddPKiS0_Pii
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM52 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM52)"
                                                                                 /* 0x001cfc00e22007f6 */
        /*0008*/                   MOV R1, c[0x0][0x20] ;                        /* 0x4c98078000870001 */
        /*0010*/                   S2R R0, SR_CTAID.X ;                          /* 0xf0c8000002570000 */
        /*0018*/                   S2R R2, SR_TID.X ;                            /* 0xf0c8000002170002 */
                                                                                 /* 0x001fd842fec20ff1 */
        /*0028*/                   XMAD.MRG R3, R0.reuse, c[0x0] [0x8].H1, RZ ;  /* 0x4f107f8000270003 */
        /*0030*/                   XMAD R2, R0.reuse, c[0x0] [0x8], R2 ;         /* 0x4e00010000270002 */
        /*0038*/                   XMAD.PSL.CBCC R0, R0.H1, R3.H1, R2 ;          /* 0x5b30011800370000 */
                                                                                 /* 0x001ff400fd4007ed */
        /*0048*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x158], PT ;  /* 0x4b6d038005670007 */
        /*0050*/                   NOP ;                                         /* 0x50b0000000070f00 */
        /*0058*/               @P0 EXIT ;                                        /* 0xe30000000000000f */
                                                                                 /* 0x081fd800fea207f1 */
        /*0068*/                   SHL R6, R0.reuse, 0x2 ;                       /* 0x3848000000270006 */
        /*0070*/                   SHR R0, R0, 0x1e ;                            /* 0x3829000001e70000 */
        /*0078*/                   IADD R4.CC, R6.reuse, c[0x0][0x140] ;         /* 0x4c10800005070604 */
                                                                                 /* 0x001fd800fe0207f2 */
        /*0088*/                   IADD.X R5, R0.reuse, c[0x0][0x144] ;          /* 0x4c10080005170005 */
        /*0090*/                   IADD R2.CC, R6, c[0x0][0x148] ;               /* 0x4c10800005270602 */
        /*0098*/                   LDG.E.CI R4, [R4] ;                           /* 0xeed4a00000070404 */
                                                                                 /* 0x001fd800f62007e2 */
        /*00a8*/                   IADD.X R3, R0, c[0x0][0x14c] ;                /* 0x4c10080005370003 */
        /*00b0*/                   LDG.E.CI R2, [R2] ;                           /* 0xeed4a00000070202 */
        /*00b8*/                   IADD R6.CC, R6, c[0x0][0x150] ;               /* 0x4c10800005470606 */
                                                                                 /* 0x001fc420fe4007f7 */
        /*00c8*/                   IADD.X R7, R0, c[0x0][0x154] ;                /* 0x4c10080005570007 */
        /*00d0*/                   IADD R0, R2, R4 ;                             /* 0x5c10000000470200 */
        /*00d8*/                   STG.E [R6], R0 ;                              /* 0xeedc200000070600 */
                                                                                 /* 0x001ffc00ffe007ea */
        /*00e8*/                   NOP ;                                         /* 0x50b0000000070f00 */
        /*00f0*/                   EXIT ;                                        /* 0xe30000000007000f */
        /*00f8*/                   BRA 0xf8 ;                                    /* 0xe2400fffff87000f */
		..........



Fatbin ptx code:
================
arch = sm_52
code version = [8,7]
host = linux
compile_size = 64bit
compressed
ptxasOptions = 
