#! /usr/bin/env bash
# export PATH="$(pwd)/build/Release/tools:$PATH"
# Determine whether the file extension of the input file is ONNX.

file=$1
file_base=${file%.*}
file_suffix=${file##*.}
model_name=$file_base
file_mlir=${file_base}.mlir

case $file_suffix in
onnx)
  iree-import-onnx $file --externalize-params --opset-version 17 -o $file_mlir
  ;;
mlir) ;;
*)
  echo "The input file must be an ONNX or mlir file."
  exit 1
  ;;
esac

file_vmfb_dist=${file_base}_dist.vmfb
file_irpa=${file_base}_params.irpa
file_print_pass=${file_base}_print_after_all.txt
file_print_pass_verbose=${file_base}_print_after_all_verbose.txt
file_print_pass_dist=${file_base}_print_after_all_dist.txt
file_print_pass_dist_verbose=${file_base}_print_after_all_dist_verbose.txt
rm -f $file_print_pass
rm -f $file_print_pass_dist
rm -f $file_print_pass_verbose
rm -f $file_print_pass_dist_verbose
rm -rf $file_dump_executable

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
  cpu)
    backend=llvm-cpu
    target_cl="--iree-llvmcpu-target-cpu=host"
    device=local-task
    file_compilation_phase=${file_base}_compilation_phase_cpu
    file_vmfb=${file_base}_cpu.vmfb
    file_dump_executable=${file_base}_dump_all_exec_cpu
    ;;
  vmvx)
    backend=vmvx
    target_cl="--iree-vmvx-enable-microkernels"
    device=local-task
    file_compilation_phase=${file_base}_compilation_phase_vmvx
    file_vmfb=${file_base}_vmvx.vmfb
    file_dump_executable=${file_base}_dump_all_exec_vmvx
    ;;
  *)
    backend=cuda
    target_cl="--iree-cuda-target=sm_80"
    device=cuda
    file_compilation_phase=${file_base}_compilation_phase_cuda
    file_vmfb=${file_base}.vmfb
    file_dump_executable=${file_base}_dump_all_exec_cuda
    ;;
  esac
  rm -rf $file_compilation_phase

  # --iree-hal-dump-executable-files-to=$file_dump_executable \
  iree-compile \
    --iree-hal-target-backends=$backend \
    $target_cl \
    --dump-compilation-phases-to=$file_compilation_phase \
    --mlir-print-ir-after-all \
    --mlir-disable-threading \
    --mlir-pass-statistics \
    $file_mlir -o $file_vmfb \
    >$file_print_pass_verbose 2>&1

  simplify_phase_name $file_compilation_phase

  python ~/.config/dotfiles/py_scripts/strip_redundant_pass.py $file_print_pass_verbose -o $file_compilation_phase/$file_print_pass

  rm -f $file_print_pass_verbose
  rm -f $file_print_pass_dist_verbose

  case $model_name in
  gpt)
    entry_function="main_graph"
    input="1x67xsi64=0"
    ;;
  softmax_regression)
    entry_function="torch_jit"
    input="1x1x28x28xf32=0"
    ;;
  esac

  iree-run-module \
    --device=$device \
    --parameters=model=$file_irpa \
    --module=$file_vmfb \
    --function=$entry_function \
    --input=$input

}

compilation_running nvgpu
# compilation_running cpu
# compilation_running vmvx
