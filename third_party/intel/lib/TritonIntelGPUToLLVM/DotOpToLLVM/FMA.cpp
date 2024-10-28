#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::array<unsigned, 3>, Value>;

namespace {
static ValueTableFMA getValueTableFromStructFMA(
    Value val, int batch, int K, int n0, int shapePerCTATile, int sizePerThread,
    ConversionPatternRewriter &rewriter, Location loc,
    TritonIntelGPUToLLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  std::cout << "    - elems.size(): " << elems.size() << std::endl;
  std::cout << "    - batch, K, n0, sizePerthread, shapePerCTATile: " << batch
            << " " << K << " " << n0 << " " << shapePerCTATile << " "
            << sizePerThread << std::endl;
  std::cout << "    - loop iterations: "
            << batch * K * n0 / shapePerCTATile * sizePerThread << std::endl;
  int index = 0;
  for (unsigned b = 0; b < batch; ++b) {
    for (unsigned k = 0; k < K; ++k) {
      for (unsigned m = 0; m < n0; m += shapePerCTATile)
        for (unsigned mm = 0; mm < sizePerThread; ++mm) {
          res[{b, m + mm, k}] = elems[index++];
        }
    }
  }
  return res;
}
} // namespace

namespace fma_details {
LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  std::cout << "  ~ convertFMADot\n";
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto bTensorTy = cast<RankedTensorType>(B.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = dLayout.getOrder();
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);
  std::cout << "    - sizePerThread: ";
  for (auto i : sizePerThread) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << "    - shapePerCTATile: ";
  for (auto i : shapePerCTATile) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << "    - order: ";
  for (auto i : order) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  size_t rank = aShapePerCTA.size();
  int batch = rank == 3 ? aShapePerCTA[0] : 1;
  int K = aShapePerCTA[rank - 1];
  int M = aShapePerCTA[rank - 2];
  int N = bShapePerCTA[rank - 1];

  int mShapePerCTATile = order[0] == rank - 1 ? shapePerCTATile[order[1]]
                                              : shapePerCTATile[order[0]];
  int mSizePerThread =
      order[0] == rank - 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile = order[0] == rank - 2 ? shapePerCTATile[order[1]]
                                              : shapePerCTATile[order[0]];
  int nSizePerThread =
      order[0] == rank - 2 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has = getValueTableFromStructFMA(llA, batch, K, M, mShapePerCTATile,
                                        mSizePerThread, rewriter, loc,
                                        typeConverter, aTensorTy);
  auto hbs = getValueTableFromStructFMA(llB, batch, K, N, nShapePerCTATile,
                                        nSizePerThread, rewriter, loc,
                                        typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == rank - 1;

  for (unsigned b = 0; b < batch; ++b) {
    for (unsigned k = 0; k < K; k++) {
      for (unsigned m = 0; m < M; m += mShapePerCTATile)
        for (unsigned n = 0; n < N; n += nShapePerCTATile)
          for (unsigned mm = 0; mm < mSizePerThread; ++mm)
            for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
              int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
              int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

              int z = isCRow
                          ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                          : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
              Type tgtTy = ret[z].getType();
              Value opA = has[{b, m + mm, k}];
              Value opB = hbs[{b, n + nn, k}];
              assert(opA.getType() == tgtTy);
              assert(opB.getType() == tgtTy);

              llvm::TypeSwitch<Type>(tgtTy)
                  .Case<FloatType>([&](auto) {
                    ret[z] =
                        rewriter.create<LLVM::FMulAddOp>(loc, opA, opB, ret[z]);
                  })
                  .Case<IntegerType>([&](auto) {
                    ret[z] = rewriter.create<LLVM::AddOp>(
                        loc, rewriter.create<LLVM::MulOp>(loc, opA, opB),
                        ret[z]);
                  });
            }
    }
  }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  std::cout << "  ~ convertFMADot end\n";

  return success();
}
} // namespace fma_details
