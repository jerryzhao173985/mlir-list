//===- ListToStandard.h - Conversion from List to Standard dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H
#define MLIR_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class Pass;
class RewritePatternSet;

void populateListToStdConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createLowerListPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LISTTOSTANDARD_LISTTOSTANDARD_H
