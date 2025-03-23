func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %N = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %op = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %op : tensor<?x?xf32>
}