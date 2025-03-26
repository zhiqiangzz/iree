// x86 CPU that has `+avx512f` feature.
#executable_target_embedded_elf_x86_64_with_encoding_layout = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "sapphirerapids", cpu_features = "+prfchw,+cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,-amx-fp8,+xsaves,+avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,+avx512ifma,+xsave,+sse4.2,+tsxldtrk,-sm3,-ptwrite,-widekl,-movrs,+invpcid,+64bit,+xsavec,-avx10.1-512,+avx512vpopcntdq,+cmov,-avx512vp2intersect,+avx512cd,+movbe,-avxvnniint8,-ccmp,+amx-int8,-kl,-avx10.1-256,+evex512,+avxvnni,-rtm,+adx,+avx2,-hreset,+movdiri,+serialize,-sha512,+vpclmulqdq,+avx512vl,-uintr,-cf,+clflushopt,-raoint,-cmpccxadd,+bmi,+amx-tile,+sse,-avx10.2-256,+gfni,-avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,+avx512f,+amx-bf16,+avx512bf16,+avx512vnni,-push2pop2,+cx8,+avx512bw,+sse3,+pku,-nf,-amx-tf32,-amx-avx512,+fsgsbase,-clzero,-mwaitx,-lwp,+lzcnt,+sha,+movdir64b,-ppx,+wbnoinvd,-enqcmd,-amx-transpose,-avx10.2-512,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,+avx512bitalg,-rdpru,+clwb,+mmx,+sse2,+rdseed,+avx512vbmi2,-prefetchi,-amx-movrs,+rdpid,-fma4,+avx512vbmi,-shstk,+vaes,+waitpkg,-sgx,+fxsr,+avx512dq,-sse4a", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>

// VMVX with ukernels enabled.
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb",
  {iree.encoding.resolver = #iree_cpu.vmvx_encoding_layout<>, ukernels = "all"}
>

util.global private @device_a = #hal.device.target<"local", {ordinal = 0 : index}, [
  #executable_target_embedded_elf_x86_64_with_encoding_layout
]> : !hal.device
util.global private @device_b = #hal.device.target<"local", {ordinal = 0 : index}, [
  #executable_target_vmvx_bytecode_fb
]> : !hal.device

func.func @foo(
  %lhs: tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>},
  %rhs: tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>}) -> (tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>}) {

  // Execute matmul on device_a and transfer the result to device_b
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %K = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %N = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %op = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %transient_op = flow.tensor.transfer %op : tensor<?x?xf32>{%M, %N} to #hal.device.affinity<@device_b>

  // Transfer input data to device_b
  %lhsb = flow.tensor.transfer %lhs : tensor<?x?xf32>{%M, %K} to #hal.device.affinity<@device_b>
  %rhsb = flow.tensor.transfer %rhs : tensor<?x?xf32>{%K, %N} to #hal.device.affinity<@device_b>
  %initb = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fillb = linalg.fill ins(%cst : f32) outs(%initb : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Execute matmul on device_b and accumulate the result and the result from device_a.
  %opb = linalg.matmul
      ins(%lhsb, %rhsb : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fillb : tensor<?x?xf32>) -> tensor<?x?xf32>
  %add = arith.addf %transient_op, %opb : tensor<?x?xf32>

  // Transfer the result from device_b -> device_a.
  %result_a = flow.tensor.transfer %add : tensor<?x?xf32>{%M, %N} to #hal.device.affinity<@device_a>

  // Return the result on device_a.
  func.return %result_a : tensor<?x?xf32>
}
