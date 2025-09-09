add_mlir_library(MLIRListToStandard
  ListToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ListProject

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRListDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
