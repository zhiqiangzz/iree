## docker dev


## Advantage of IREE
1. [documents](https://iree.dev/)
2. [phase recovry](https://iree.dev/developers/general/developer-tips/#compiling-phase-by-phase) 
```shell
iree-compile --compile-to=abi xxx.mlir
modify 'xxx.mlir' mannually
iree-compile --compile-from=abi xxx.bin
```
3. trace 
```shell
iree-run-module \
  --device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d \
  --device=cuda://GPU-fe10e8dc-a093-1f35-4594-b738ceab830b \
  --module=dt_multi_device_gpu_gpu.vmfb \
  --function=foo \
  --input=1x10xf32=2 \
  --input=10x1xf32=4 \
  --trace_execution=true
```
4. [community](https://discord.gg/wEWh6Z9nMU)
5. similar to byteir main components
	- frontend onnx/ [torch(iree-turbine)](https://iree.dev/guides/ml-frameworks/pytorch/)/ tensorflow(stableHLO)
	- compiler --iree-compile
	- runtime --iree-run-module
```shell
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e build/Release/compiler
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e build/Release/runtime
```

## homogeneous multi-device whin one node
1. [llama e2e compilation and serving](https://github.com/nod-ai/shark-ai/blob/main/docs/shortfin/llm/user/llama_serving.md#compiling-to-vmfb)
Unfortunately, SHARK-AI currently only supports multi-GPU inference and deployment on AMD GPUs.
[shark-ai model serving(based on iree-toolchain) developed by nod-ai which affiliates to AMD](https://github.com/nod-ai/shark-ai/tree/main)
[cuda](https://github.com/iree-org/iree/issues/20308)
2. [nccl](/home/zhiqiangz/projects/mlc/iree/runtime/src/iree/hal/drivers/cuda/nccl_dynamic_symbols.h)


## Compilation phases

## async exec model
1. dispatch inner sync(barrier)/ sync between different dispatch region(fence)
```mlir
// device_b
%ref_17 = vm.call @hal.command_buffer.create(%device_b, %c1, %c3, %c-1_1, %zero) : (!vm.ref<!hal.device>, i32, i32, i64, i32) -> !vm.ref<!hal.command_buffer>
vm.call @hal.command_buffer.copy_buffer(%ref_17, %zero, %zero, %ref, %zero_0, %ref_15, %zero_0, %3) : (!vm.ref<!hal.command_buffer>, i32, i32, !vm.ref<!hal.buffer>, i64, !vm.ref<!hal.buffer>, i64, i64) -> ()
vm.call @hal.command_buffer.copy_buffer(%ref_17, %zero, %zero, %ref_3, %zero_0, %ref_15, %24, %7) : (!vm.ref<!hal.command_buffer>, i32, i32, !vm.ref<!hal.buffer>, i64, !vm.ref<!hal.buffer>, i64, i64) -> ()
vm.call @hal.command_buffer.execution_barrier(%ref_17, %c28, %c13, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
vm.call.variadic @hal.command_buffer.dispatch(%ref_17, %__device_b_executable_0_dt_multi_device_gpu_gpu_linked, %c1, %20, %17, %c1, %zero_0, [%29, %31, %32, %34, %11, %13, %17, %19, %20, %22], [(%zero, %zero, %ref_15, %zero_0, %28), (%zero, %zero, %ref_5, %zero_0, %8), (%zero, %zero, %ref_15, %zero_0, %28)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32, i64, i32 ..., tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64> ...)
vm.call @hal.command_buffer.execution_barrier(%ref_17, %c28, %c13, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
vm.call @hal.command_buffer.copy_buffer(%ref_17, %zero, %zero, %ref_15, %27, %ref_13, %zero_0, %8) : (!vm.ref<!hal.command_buffer>, i32, i32, !vm.ref<!hal.buffer>, i64, !vm.ref<!hal.buffer>, i64, i64) -> ()
vm.call @hal.command_buffer.execution_barrier(%ref_17, %c28, %c13, %zero) : (!vm.ref<!hal.command_buffer>, i32, i32, i32) -> ()
vm.call @hal.command_buffer.finalize(%ref_17) : (!vm.ref<!hal.command_buffer>) -> ()
%ref_18 = vm.call @hal.fence.create(%device_b, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.fence>
vm.call.variadic @hal.device.queue.execute(%device_b, %c-1_1, %ref_16, %ref_18, [%ref_17]) : (!vm.ref<!hal.device>, i64, !vm.ref<!hal.fence>, !vm.ref<!hal.fence>, !vm.ref<!hal.command_buffer> ...)
%ref_19 = vm.call @hal.fence.create(%device_b, %zero) : (!vm.ref<!hal.device>, i32) -> !vm.ref<!hal.fence>
vm.call @hal.device.queue.dealloca(%device_b, %c-1_1, %ref_18, %ref_19, %ref_15) : (!vm.ref<!hal.device>, i64, !vm.ref<!hal.fence>, !vm.ref<!hal.fence>, !vm.ref<!hal.buffer>) -> ()
```

> I can't find the prior right now, but there was an old paper from Google that used RL to derive device assignments by manipulating transfers in the graph.
> I don't think it's a compiler problem really: it is almost trivial to do by manipulating the pytorch module graph before any compiler sees it. And that would also allow for more interesting methods for determining placement that what you would get out of a lower level compiler (ie. You can apply ml algorithms or heuristics to the high level graph).
> When we're working on optimizing a model so that it is as good as possible for some hardware, we just work at the model source level and write it how we want it. Especially when trying to exploit new hardware, there is just no other way that is effective, as the automated tools are not good for pathfinding. 
> Then if you find a recipe or transformation that has wide applicability to a family of models, then you can automate that with some form of pytorch graph/module transform or a lower level compiler/op level transform.

finer granularity computation/communication overlap?


## New arch support
1. HAL(high abstract layer)[https://iree.dev/reference/mlir-dialects/HAL/]
Implement corresponding hook(virtual function override)
-  device instance/register
-  memory management(allocation)
-  data transfer(data encoding/layout)
-  sync semaphore/barrier/await
-  command exec model(cuda stream/cuda graph...)
-  executable bin(lib/ptx?/kernel)

## Heterogeneous system test
1. cpu+vmvx and 2nvgpu test
```shell
## cpu and vmvx(iree virtual machine and vm-based linear algebra) 
cd multiple_devices
./multi_device_test.sh dt_multi_device_cpu_vmvx.mlir
## cuda[0] and cuda[1]
./multi_device_test.sh dt_multi_device_gpu_gpu.mlir
```

2. gpu+pim async heterogeneous?
## 