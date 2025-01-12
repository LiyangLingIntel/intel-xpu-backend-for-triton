#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

static Value mxfpScaleBf16(ConversionPatternRewriter &rewriter, Location loc,
                           Value v, Value scale) {
  Value vBf16 = bitcast(v, bf16_ty);
  Value nanBf16 = bitcast(i16_val(0x7fff), bf16_ty);
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value scaleBf16 = bitcast(shl(zext(i16_ty, scale), i16_val(7)), bf16_ty);

  Value v0 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, vBf16);
  Value v1 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, scaleBf16);
  auto result = rewriter.create<LLVM::FMulOp>(loc, f32_ty, v0, v1);
  auto undefRounding = static_cast<mlir::triton::RoundingMode>(-1);
  Value scaledBf16 = mlir::triton::intel::convertFp32ToBf16(
      loc, rewriter, result, undefRounding);
  // Account for NaN in the scale as per the mxfp specification.
  return select(scaleIsNan, nanBf16, scaledBf16);
};

class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<UpcastMXFPOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto operands = adaptor.getOperands();
    SmallVector<Value> xVals = unpackLLElements(loc, operands[0], rewriter);
    SmallVector<Value> scaleVals = unpackLLElements(loc, operands[1], rewriter);
    ScaleDotElemType fpType = op.getFpType();

    Value tid = tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    if (fpType == ScaleDotElemType::E2M1)
      xVals = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);

    auto xType = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto dotEnc = cast<DotOperandEncodingAttr>(xType.getEncoding());
    // For RHS dot operand with dot layout encoding, each thread access tensor
    // elements in column which require scaling values from threads across
    // warps. We can only access the scaling values by shuffle in the same warp.
    assert(dotEnc.getOpIdx() == 0 && "NYI: rhs scale with dot encoding");

    auto dpasEnc = cast<DpasEncodingAttr>(dotEnc.getParent());
    unsigned instShapeM = dpasEnc.getDPASInstShapeA()[0];
    unsigned instShapeK = dpasEnc.getDPASInstShapeA()[1];
    constexpr unsigned scalingBlockSize = 32;
    unsigned repSize = scalingBlockSize / instShapeK;
    unsigned subTileSize = instShapeM;
    // kWidth here is the contiguous number of elements each thread access.
    unsigned kWidth = dpasEnc.getOpsPerChannel() / 2;
    unsigned numMxfp =
        TritonGPUDialect::TritonGPUDialect::getThreadsPerWarp(mod) / instShapeM;
    unsigned mxfpSize = repSize * subTileSize * kWidth;
    constexpr unsigned numScales = 16;

    if (fpType == ScaleDotElemType::E2M1) {
      repSize /= 2; // for E2M1, actual scaling block size is 16
      kWidth *= 2;  // 2 fp4 are packed in one i8
    }

    Value c = udiv(laneId, i32_val(numScales));
    SmallVector<Value, 16> ci;
    for (int row = 0; row < numMxfp; ++row)
      for (int col = 0; col < subTileSize; ++col)
        ci.emplace_back(add(c, i32_val(row + 2 * col)));

    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      for (int mxfp = 0; mxfp < numMxfp; ++mxfp) {
        SmallVector<Value, 8> si;
        for (int subTile = 0; subTile < 8; ++subTile)
          si.emplace_back(targetInfo.shuffleIdx(rewriter, loc, scaleVal,
                                                ci[8 * mxfp + subTile]));
        for (int rep = 0; rep < repSize; ++rep) {
          for (int subTile = 0; subTile < subTileSize; ++subTile) {
            for (int k = 0; k < kWidth; ++k) {
              unsigned idx = i * scalingBlockSize + mxfp * mxfpSize +
                             rep * subTileSize * kWidth + subTile * kWidth + k;
              xVals[idx] =
                  mxfpScaleBf16(rewriter, loc, xVals[idx], si[subTile]);
            }
          }
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::intel::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
