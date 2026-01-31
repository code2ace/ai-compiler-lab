; ModuleID = 'lowered.ll'
source_filename = "mat4x4_16acc_test.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

%"struct.std::__1::array" = type { [16 x float] }
%"struct.std::__1::__less" = type { i8 }

@.str = private unnamed_addr constant [30 x i8] c"[%s] max|diff| = %.8f  => %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"OK\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"MISMATCH\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"A\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"C_ref\00", align 1
@.str.6 = private unnamed_addr constant [6 x i8] c"C_ker\00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c"%s =\0A\00", align 1
@.str.8 = private unnamed_addr constant [3 x i8] c"  \00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"%8.4f \00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define void @mat4x4_16acc_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C) #0 {
entry:
  %arow0 = load <4 x float>, ptr %A, align 16
  %Arow1 = getelementptr inbounds nuw i8, ptr %A, i64 16
  %arow1 = load <4 x float>, ptr %Arow1, align 16
  %0 = shufflevector <4 x float> %arow0, <4 x float> %arow1, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %Arow2 = getelementptr inbounds nuw i8, ptr %A, i64 32
  %arow2 = load <4 x float>, ptr %Arow2, align 16
  %Arow3 = getelementptr inbounds nuw i8, ptr %A, i64 48
  %arow3 = load <4 x float>, ptr %Arow3, align 16
  %1 = shufflevector <4 x float> %arow2, <4 x float> %arow3, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  %2 = shufflevector <4 x float> %0, <4 x float> %1, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %BRow3for.inc.clone = getelementptr inbounds nuw i8, ptr %B, i64 48
  %brow3for.inc.clone = load <4 x float>, ptr %BRow3for.inc.clone, align 16
  %3 = shufflevector <4 x float> %brow3for.inc.clone, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %4 = shufflevector <4 x float> %0, <4 x float> %1, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %BRow2for.inc.clone = getelementptr inbounds nuw i8, ptr %B, i64 32
  %brow2for.inc.clone = load <4 x float>, ptr %BRow2for.inc.clone, align 16
  %5 = shufflevector <4 x float> %brow2for.inc.clone, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %6 = shufflevector <4 x float> %arow0, <4 x float> %arow1, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %7 = shufflevector <4 x float> %arow2, <4 x float> %arow3, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  %8 = shufflevector <4 x float> %6, <4 x float> %7, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %BRow1for.inc.clone = getelementptr inbounds nuw i8, ptr %B, i64 16
  %brow1for.inc.clone = load <4 x float>, ptr %BRow1for.inc.clone, align 16
  %9 = shufflevector <4 x float> %brow1for.inc.clone, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %10 = shufflevector <4 x float> %6, <4 x float> %7, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %brow0for.inc.clone = load <4 x float>, ptr %B, align 16
  %11 = shufflevector <4 x float> %brow0for.inc.clone, <4 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %12 = fmul fast <4 x float> %10, %11
  %13 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %8, <4 x float> %9, <4 x float> %12)
  %14 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %4, <4 x float> %5, <4 x float> %13)
  %15 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %2, <4 x float> %3, <4 x float> %14)
  %16 = extractelement <4 x float> %15, i64 3
  %17 = shufflevector <4 x float> %brow3for.inc.clone, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %18 = shufflevector <4 x float> %brow2for.inc.clone, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %19 = shufflevector <4 x float> %brow1for.inc.clone, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %20 = shufflevector <4 x float> %brow0for.inc.clone, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %21 = fmul fast <4 x float> %10, %20
  %22 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %8, <4 x float> %19, <4 x float> %21)
  %23 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %4, <4 x float> %18, <4 x float> %22)
  %24 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %2, <4 x float> %17, <4 x float> %23)
  %25 = extractelement <4 x float> %24, i64 3
  %26 = shufflevector <4 x float> %brow3for.inc.clone, <4 x float> poison, <4 x i32> zeroinitializer
  %27 = shufflevector <4 x float> %brow2for.inc.clone, <4 x float> poison, <4 x i32> zeroinitializer
  %28 = shufflevector <4 x float> %brow1for.inc.clone, <4 x float> poison, <4 x i32> zeroinitializer
  %29 = shufflevector <4 x float> %brow0for.inc.clone, <4 x float> poison, <4 x i32> zeroinitializer
  %30 = fmul fast <4 x float> %10, %29
  %31 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %8, <4 x float> %28, <4 x float> %30)
  %32 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %4, <4 x float> %27, <4 x float> %31)
  %33 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %2, <4 x float> %26, <4 x float> %32)
  %34 = extractelement <4 x float> %33, i64 0
  %35 = shufflevector <4 x float> %brow3for.inc.clone, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %36 = shufflevector <4 x float> %brow2for.inc.clone, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %37 = shufflevector <4 x float> %brow1for.inc.clone, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %38 = shufflevector <4 x float> %brow0for.inc.clone, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %39 = fmul fast <4 x float> %10, %38
  %40 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %8, <4 x float> %37, <4 x float> %39)
  %41 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %4, <4 x float> %36, <4 x float> %40)
  %42 = call fast <4 x float> @llvm.fma.v4f32(<4 x float> %2, <4 x float> %35, <4 x float> %41)
  %43 = extractelement <4 x float> %42, i64 0
  %44 = extractelement <4 x float> %24, i64 0
  %45 = extractelement <4 x float> %15, i64 0
  %46 = extractelement <4 x float> %33, i64 1
  %47 = extractelement <4 x float> %42, i64 1
  %48 = extractelement <4 x float> %24, i64 1
  %49 = extractelement <4 x float> %15, i64 1
  %50 = extractelement <4 x float> %33, i64 2
  %51 = extractelement <4 x float> %42, i64 2
  %52 = extractelement <4 x float> %24, i64 2
  %53 = extractelement <4 x float> %15, i64 2
  %54 = extractelement <4 x float> %33, i64 3
  %55 = extractelement <4 x float> %42, i64 3
  store float %34, ptr %C, align 4
  %arrayidx58 = getelementptr inbounds nuw i8, ptr %C, i64 4
  store float %43, ptr %arrayidx58, align 4
  %arrayidx59 = getelementptr inbounds nuw i8, ptr %C, i64 8
  store float %44, ptr %arrayidx59, align 4
  %arrayidx60 = getelementptr inbounds nuw i8, ptr %C, i64 12
  store float %45, ptr %arrayidx60, align 4
  %arrayidx61 = getelementptr inbounds nuw i8, ptr %C, i64 16
  store float %46, ptr %arrayidx61, align 4
  %arrayidx62 = getelementptr inbounds nuw i8, ptr %C, i64 20
  store float %47, ptr %arrayidx62, align 4
  %arrayidx63 = getelementptr inbounds nuw i8, ptr %C, i64 24
  store float %48, ptr %arrayidx63, align 4
  %arrayidx64 = getelementptr inbounds nuw i8, ptr %C, i64 28
  store float %49, ptr %arrayidx64, align 4
  %arrayidx65 = getelementptr inbounds nuw i8, ptr %C, i64 32
  store float %50, ptr %arrayidx65, align 4
  %arrayidx66 = getelementptr inbounds nuw i8, ptr %C, i64 36
  store float %51, ptr %arrayidx66, align 4
  %arrayidx67 = getelementptr inbounds nuw i8, ptr %C, i64 40
  store float %52, ptr %arrayidx67, align 4
  %arrayidx68 = getelementptr inbounds nuw i8, ptr %C, i64 44
  store float %53, ptr %arrayidx68, align 4
  %arrayidx69 = getelementptr inbounds nuw i8, ptr %C, i64 48
  store float %54, ptr %arrayidx69, align 4
  %arrayidx70 = getelementptr inbounds nuw i8, ptr %C, i64 52
  store float %55, ptr %arrayidx70, align 4
  %arrayidx71 = getelementptr inbounds nuw i8, ptr %C, i64 56
  store float %25, ptr %arrayidx71, align 4
  %arrayidx72 = getelementptr inbounds nuw i8, ptr %C, i64 60
  store float %16, ptr %arrayidx72, align 4
  ret void
}

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define void @run_case(ptr noundef %label, ptr noundef nonnull align 4 dereferenceable(64) %A, ptr noundef nonnull align 4 dereferenceable(64) %B, i1 noundef zeroext %verbose) #1 {
entry:
  %C_ref = alloca %"struct.std::__1::array", align 4
  %C_ker = alloca %"struct.std::__1::array", align 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %C_ref, i8 0, i64 64, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %C_ker, i8 0, i64 64, i1 false)
  %call = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %A) #7
  %call1 = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %B) #7
  %call2 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ref) #7
  call void @_ZL17mat4x4_ref_triplePKfS0_Pf(ptr noundef %call, ptr noundef %call1, ptr noundef %call2)
  %call3 = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %A) #7
  %call4 = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %B) #7
  %call5 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ker) #7
  call void @mat4x4_16acc_kernel(ptr noundef %call3, ptr noundef %call4, ptr noundef %call5)
  %call6 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ref) #7
  %call7 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ker) #7
  %call8 = call noundef float @_ZL12max_abs_diffPKfS0_(ptr noundef %call6, ptr noundef %call7)
  %call9 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ref) #7
  %call10 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ker) #7
  %call11 = call noundef zeroext i1 @_ZL16nearly_equal_matPKfS0_f(ptr noundef %call9, ptr noundef %call10, float noundef 0x3EE4F8B580000000)
  %conv = fpext float %call8 to double
  %cond = select i1 %call11, ptr @.str.1, ptr @.str.2
  %call13 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %label, double noundef %conv, ptr noundef nonnull %cond)
  %call11.not = xor i1 %call11, true
  %brmerge = or i1 %verbose, %call11.not
  br i1 %brmerge, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call16 = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %A) #7
  call void @_ZL9print_matPKcPKf(ptr noundef nonnull @.str.3, ptr noundef %call16)
  %call17 = call noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %B) #7
  call void @_ZL9print_matPKcPKf(ptr noundef nonnull @.str.4, ptr noundef %call17)
  %call18 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ref) #7
  call void @_ZL9print_matPKcPKf(ptr noundef nonnull @.str.5, ptr noundef %call18)
  %call19 = call noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %C_ker) #7
  call void @_ZL9print_matPKcPKf(ptr noundef nonnull @.str.6, ptr noundef %call19)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  br i1 %call11, label %if.end22, label %if.then21

if.then21:                                        ; preds = %if.end
  call void @exit(i32 noundef 1) #8
  unreachable

if.end22:                                         ; preds = %if.end
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define internal void @_ZL17mat4x4_ref_triplePKfS0_Pf(ptr noundef %A, ptr noundef %B, ptr noundef %C) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc20, %entry
  %indvars.iv10 = phi i64 [ %indvars.iv.next11, %for.inc20 ], [ 0, %entry ]
  %exitcond16.not = icmp eq i64 %indvars.iv10, 4
  br i1 %exitcond16.not, label %for.end22, label %for.cond1

for.cond1:                                        ; preds = %for.cond, %for.end
  %indvars.iv5 = phi i64 [ %indvars.iv.next6, %for.end ], [ 0, %for.cond ]
  %exitcond9.not = icmp eq i64 %indvars.iv5, 4
  br i1 %exitcond9.not, label %for.inc20, label %for.cond4

for.cond4:                                        ; preds = %for.cond1, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.cond1 ]
  %cij.0 = phi float [ %add12, %for.inc ], [ 0.000000e+00, %for.cond1 ]
  %exitcond.not = icmp eq i64 %indvars.iv, 4
  br i1 %exitcond.not, label %for.end, label %for.inc

for.inc:                                          ; preds = %for.cond4
  %.idx1 = shl nuw nsw i64 %indvars.iv10, 4
  %0 = getelementptr inbounds nuw i8, ptr %A, i64 %.idx1
  %arrayidx = getelementptr inbounds nuw float, ptr %0, i64 %indvars.iv
  %1 = load float, ptr %arrayidx, align 4
  %.idx2 = shl nuw nsw i64 %indvars.iv, 4
  %2 = getelementptr inbounds nuw i8, ptr %B, i64 %.idx2
  %arrayidx10 = getelementptr inbounds nuw float, ptr %2, i64 %indvars.iv5
  %3 = load float, ptr %arrayidx10, align 4
  %mul11 = fmul float %1, %3
  %add12 = fadd float %cij.0, %mul11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond4, !llvm.loop !6

for.end:                                          ; preds = %for.cond4
  %.idx = shl nuw nsw i64 %indvars.iv10, 4
  %4 = getelementptr inbounds nuw i8, ptr %C, i64 %.idx
  %arrayidx16 = getelementptr inbounds nuw float, ptr %4, i64 %indvars.iv5
  store float %cij.0, ptr %arrayidx16, align 4
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  br label %for.cond1, !llvm.loop !8

for.inc20:                                        ; preds = %for.cond1
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  br label %for.cond, !llvm.loop !9

for.end22:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define linkonce_odr hidden noundef ptr @_ZNKSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %this) #0 {
entry:
  ret ptr %this
}

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define linkonce_odr hidden noundef ptr @_ZNSt3__15arrayIfLm16EE4dataB8se210000Ev(ptr noundef nonnull align 4 dereferenceable(64) %this) #0 {
entry:
  ret ptr %this
}

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define internal noundef float @_ZL12max_abs_diffPKfS0_(ptr noundef %X, ptr noundef %Y) #1 {
entry:
  %m = alloca float, align 4
  %ref.tmp = alloca float, align 4
  store float 0.000000e+00, ptr %m, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %exitcond.not = icmp eq i64 %indvars.iv, 16
  br i1 %exitcond.not, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds nuw float, ptr %X, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw float, ptr %Y, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %sub = fsub float %0, %1
  %call = call noundef float @_ZNSt3__16__math4fabsB8se210000Ef(float noundef %sub) #7
  store float %call, ptr %ref.tmp, align 4
  %call3 = call noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3__13maxB8se210000IfEERKT_S3_S3_(ptr noundef nonnull align 4 dereferenceable(4) %m, ptr noundef nonnull align 4 dereferenceable(4) %ref.tmp)
  %2 = load float, ptr %call3, align 4
  store float %2, ptr %m, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !10

for.end:                                          ; preds = %for.cond
  %3 = load float, ptr %m, align 4
  ret float %3
}

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define internal noundef zeroext i1 @_ZL16nearly_equal_matPKfS0_f(ptr noundef %X, ptr noundef %Y, float noundef %eps) #1 {
entry:
  %call = call noundef float @_ZL12max_abs_diffPKfS0_(ptr noundef %X, ptr noundef %Y)
  %cmp = fcmp ole float %call, %eps
  ret i1 %cmp
}

declare i32 @printf(ptr noundef, ...) #3

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define internal void @_ZL9print_matPKcPKf(ptr noundef %name, ptr noundef %M) #1 {
entry:
  %call = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef %name)
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.end ], [ 0, %entry ]
  %exitcond7.not = icmp eq i64 %indvars.iv3, 4
  br i1 %exitcond7.not, label %for.end9, label %for.body

for.body:                                         ; preds = %for.cond
  %call1 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8)
  br label %for.cond2

for.cond2:                                        ; preds = %for.body4, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 0, %for.body ]
  %exitcond.not = icmp eq i64 %indvars.iv, 4
  br i1 %exitcond.not, label %for.end, label %for.body4

for.body4:                                        ; preds = %for.cond2
  %.idx = shl nuw nsw i64 %indvars.iv3, 4
  %0 = getelementptr inbounds nuw i8, ptr %M, i64 %.idx
  %arrayidx = getelementptr inbounds nuw float, ptr %0, i64 %indvars.iv
  %1 = load float, ptr %arrayidx, align 4
  %conv = fpext float %1 to double
  %call5 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, double noundef %conv)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond2, !llvm.loop !11

for.end:                                          ; preds = %for.cond2
  %putchar = call i32 @putchar(i32 10)
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond, !llvm.loop !12

for.end9:                                         ; preds = %for.cond
  ret void
}

; Function Attrs: noreturn
declare void @exit(i32 noundef) #4

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define linkonce_odr hidden noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3__13maxB8se210000IfEERKT_S3_S3_(ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b) #1 {
entry:
  %call = call noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_(ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b)
  ret ptr %call
}

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define linkonce_odr hidden noundef float @_ZNSt3__16__math4fabsB8se210000Ef(float noundef %__x) #0 {
entry:
  %0 = call float @llvm.fabs.f32(float %__x)
  ret float %0
}

; Function Attrs: mustprogress noinline ssp uwtable(sync)
define linkonce_odr hidden noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3__13maxB8se210000IfNS_6__lessIvvEEEERKT_S5_S5_T0_(ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b) #1 {
entry:
  %__comp = alloca %"struct.std::__1::__less", align 1
  %call = call noundef zeroext i1 @_ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %__comp, ptr noundef nonnull align 4 dereferenceable(4) %__a, ptr noundef nonnull align 4 dereferenceable(4) %__b)
  %__b.__a = select i1 %call, ptr %__b, ptr %__a
  ret ptr %__b.__a
}

; Function Attrs: mustprogress noinline nounwind ssp uwtable(sync)
define linkonce_odr hidden noundef zeroext i1 @_ZNKSt3__16__lessIvvEclB8se210000IffEEbRKT_RKT0_(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef nonnull align 4 dereferenceable(4) %__lhs, ptr noundef nonnull align 4 dereferenceable(4) %__rhs) #0 {
entry:
  %0 = load float, ptr %__lhs, align 4
  %1 = load float, ptr %__rhs, align 4
  %cmp = fcmp olt float %0, %1
  ret i1 %cmp
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #5

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) #5

attributes #0 = { mustprogress noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { mustprogress noinline ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #4 = { noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nofree nounwind }
attributes #7 = { nounwind }
attributes #8 = { cold noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 21.0.0git (https://github.com/riptio777/llvm-project.git 0053de4c88f93e41709dccc38d1df1946b1197cc)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
