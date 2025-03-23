module {
  util.global private @net.1.weight = #stream.parameter.named<"model"::"net.1.weight"> : tensor<10x784xf32>
  func.func @torch_jit(%arg0: !torch.vtensor<[?,1,28,28],f32>) -> !torch.vtensor<[?,10],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.0.0"} {
    %net.1.weight = util.global.load @net.1.weight : tensor<10x784xf32>
    %0 = torch_c.from_builtin_tensor %net.1.weight : tensor<10x784xf32> -> !torch.vtensor<[10,784],f32>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_net.1.bias> : tensor<10xf32>} : () -> !torch.vtensor<[10],f32> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[?,1,28,28],f32>) -> !torch.vtensor<[?,784],f32> 
    %3 = torch.operator "onnx.Gemm"(%2, %0, %1) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[?,784],f32>, !torch.vtensor<[10,784],f32>, !torch.vtensor<[10],f32>) -> !torch.vtensor<[?,10],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.Add"(%3, %4) : (!torch.vtensor<[?,10],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[?,10],f32> 
    %6 = torch.operator "onnx.LogSoftmax"(%5) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[?,10],f32>) -> !torch.vtensor<[?,10],f32> 
    return %6 : !torch.vtensor<[?,10],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _net.1.bias: "0x08000000C96403BBDC26D2BB7E76A0BDF186BEBAD1AC5ABE8972163F53A2ED3DEDE4D13B61C9F3BD3DA03CBE",
      _: "0x08000000CD4C6943"
    }
  }
#-}

