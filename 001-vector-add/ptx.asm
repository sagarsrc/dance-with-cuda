
Fatbin elf code:
================
arch = sm_52
code version = [1,7]
host = linux
compile_size = 64bit

Fatbin elf code:
================
arch = sm_52
code version = [1,7]
host = linux
compile_size = 64bit

Fatbin ptx code:
================
arch = sm_52
code version = [8,7]
host = linux
compile_size = 64bit
compressed
ptxasOptions = 

//
//
//
//
//
//

.version 8.7
.target sm_52
.address_size 64

//

.visible .entry _Z9vectorAddPKiS0_Pii(
.param .u64 _Z9vectorAddPKiS0_Pii_param_0,
.param .u64 _Z9vectorAddPKiS0_Pii_param_1,
.param .u64 _Z9vectorAddPKiS0_Pii_param_2,
.param .u32 _Z9vectorAddPKiS0_Pii_param_3
)
{
.reg .pred %p<2>;
.reg .b32 %r<9>;
.reg .b64 %rd<11>;


ld.param.u64 %rd1, [_Z9vectorAddPKiS0_Pii_param_0];
ld.param.u64 %rd2, [_Z9vectorAddPKiS0_Pii_param_1];
ld.param.u64 %rd3, [_Z9vectorAddPKiS0_Pii_param_2];
ld.param.u32 %r2, [_Z9vectorAddPKiS0_Pii_param_3];
mov.u32 %r3, %ctaid.x;
mov.u32 %r4, %ntid.x;
mov.u32 %r5, %tid.x;
mad.lo.s32 %r1, %r3, %r4, %r5;
setp.ge.s32 %p1, %r1, %r2;
@%p1 bra $L__BB0_2;

cvta.to.global.u64 %rd4, %rd1;
mul.wide.s32 %rd5, %r1, 4;
add.s64 %rd6, %rd4, %rd5;
cvta.to.global.u64 %rd7, %rd2;
add.s64 %rd8, %rd7, %rd5;
ld.global.nc.u32 %r6, [%rd8];
ld.global.nc.u32 %r7, [%rd6];
add.s32 %r8, %r6, %r7;
cvta.to.global.u64 %rd9, %rd3;
add.s64 %rd10, %rd9, %rd5;
st.global.u32 [%rd10], %r8;

$L__BB0_2:
ret;

}


