#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.target<arch = "sm_80", features = "+ptx76", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<NV_WMMA_F32_16x16x16_F16>, <NV_WMMA_F16_16x16x16_F16>], subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 166912, max_workgroup_counts = [2147483647, 65535, 65535]>>}>

#device_target_cuda_0_ = #hal.device.target<"cuda", {ordinal = 0 : index}, [#executable_target_cuda_nvptx_fb]> : !hal.device
#device_target_cuda_1_ = #hal.device.target<"cuda", {ordinal = 0 : index}, [#executable_target_cuda_nvptx_fb]> : !hal.device

util.global private @device_a = #device_target_cuda_0_
util.global private @device_b = #device_target_cuda_1_

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
