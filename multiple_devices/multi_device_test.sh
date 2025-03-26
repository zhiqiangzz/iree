#! /usr/bin/env bash
# export PATH="$(pwd)/../build/Release/tools:$PATH"
# Determine whether the file extension of the input file is ONNX.

file=$1
option=$2
case "$option" in
"" | "run" | "compile") ;;
*)
  echo "Please input the correct option at \$2: None or run or compile"
  exit 1
  ;;
esac

file_base=${file%.*}
device_select=$(echo $file_base | awk -F'_' '{print $(NF-1)"_"$(NF)}')
file_suffix=${file##*.}
model_name=$file_base
file_mlir=${file_base}.mlir

case $file_suffix in
# onnx)
#   iree-import-onnx $file --externalize-params --opset-version 17 -o $file_mlir
#   ;;
mlir) ;;
*)
  echo "The input file must be an ONNX or mlir file."
  exit 1
  ;;
esac

file_vmfb_dist=${file_base}_dist.vmfb
file_irpa=${file_base}_params.irpa
file_print_pass=${file_base}_print_after_all_heterogeneous.txt
file_print_pass_verbose=${file_base}_print_after_all_heterogeneous_verbose.txt
# file_print_pass_dist=${file_base}_print_after_all_dist.txt
# file_print_pass_dist_verbose=${file_base}_print_after_all_dist_verbose.txt
rm -f $file_print_pass
rm -f $file_print_pass_verbose
# rm -f $file_print_pass_dist
# rm -f $file_print_pass_dist_verbose

# # multiple device
# iree-compile \
#   --iree-hal-target-device='cuda[0]' \
#   --iree-hal-target-device='cuda[1]' \
#   --iree-cuda-target=sm_80 \
#   --mlir-print-ir-after-all \
#   --mlir-disable-threading \
#   --mlir-pass-statistics \
#   $file_mlir -o $file_vmfb_dist \
#   >$file_print_pass_dist_verbose 2>&1

# python ~/.config/dotfiles/py_scripts/strip_redundant_pass.py $file_print_pass_dist_verbose -o $file_print_pass_dist

function simplify_phase_name() {
  pushd $1 >/dev/null
  for file in *; do
    if [ -f "$file" ]; then
      newname=$(echo "$file" | sed 's/^[^0-9]*//')
      mv "$file" "$newname"
    fi
  done
  popd >/dev/null
}

function compilation_running() {
  case $1 in
  llvm_cpu-vmvx)
    compile_cl="--iree-llvmcpu-target-cpu=host"
    run_cl="--device=local-task"
    ;;
  llvm_cpu-gpu)
    compile_cl="--iree-llvmcpu-target-cpu=host"
    run_cl="--device=local-task --device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d"
    ;;
  gpu-gpu)
    compile_cl="--iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=cuda[0] \
    --iree-hal-target-device=cuda[1] \
    --iree-cuda-target=sm_80"
    run_cl="--device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d \
    --device=cuda://GPU-fe10e8dc-a093-1f35-4594-b738ceab830b \
    --trace_execution=true"
    # run_cl="--device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d"
    # run_cl="--device=cuda"
    ;;
  gpu-dup)
    compile_cl="--iree-llvmcpu-target-cpu=host \
    --iree-cuda-target=sm_80"
    run_cl="--device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d"
    ;;
  # 2-gpus)
  #   compile_cl="--iree-llvmcpu-target-cpu=host \
  #   --iree-hal-target-device=cuda[0] \
  #   --iree-hal-target-device=cuda[1] \
  #   --iree-cuda-target=sm_80"
  #   run_cl="--device=cuda://GPU-55b2b321-c6c9-dc31-afe3-72f56fb9612d --device=cuda://GPU-fe10e8dc-a093-1f35-4594-b738ceab830b"
  #   ;;
  *)
    echo "Please input multi device mlir program"
    exit 1
    ;;
  esac
  file_compilation_phase=${file_base}_compilation_phase
  file_vmfb=${file_base}.vmfb
  file_dump_executable=${file_base}_dump_all_exec

  if [ -z "$option" ] || [ $option = "compile" ]; then
    rm -rf $file_compilation_phase
    # --iree-hal-dump-executable-files-to=$file_dump_executable \
    iree-compile \
      $compile_cl \
      --dump-compilation-phases-to=$file_compilation_phase \
      --mlir-print-ir-after-all \
      --mlir-disable-threading \
      --mlir-pass-statistics \
      $file_mlir -o $file_vmfb \
      >$file_print_pass_verbose 2>&1

    simplify_phase_name $file_compilation_phase

    python ~/.config/dotfiles/py_scripts/strip_redundant_pass.py $file_print_pass_verbose -o $file_compilation_phase/$file_print_pass

    rm -f $file_print_pass_verbose
    # rm -f $file_print_pass_dist_verbose
  fi

  if [ -z "$option" ] || [ $option = "run" ]; then
    case $model_name in
    dt_multi_device_cpu_vmvx | dt_multi_device_gpu_gpu | dt_multi_device_cpu_gpu | dt_multi_device_gpu_dup)
      entry_function="foo"
      input="--input=1x10xf32=2 --input=10x1xf32=4"
      ;;
    *)
      echo "currently dont't support test $model_name"
      exit 1
      ;;
    esac

    # --parameters=model=$file_irpa \
    iree-run-module \
      $run_cl \
      --module=$file_vmfb \
      --function=$entry_function \
      $input
  fi
}

case $device_select in
cpu_vmvx)
  compilation_running llvm_cpu-vmvx
  ;;
cpu_gpu)
  compilation_running llvm_cpu-gpu
  ;;
gpu_gpu)
  compilation_running gpu-gpu
  ;;
gpu_dup)
  compilation_running gpu-dup
  ;;
2_gpus)
  compilation_running 2-gpus
  ;;
*)
  echo "Don't support such multi-device"
  exit 1
  ;;
esac
