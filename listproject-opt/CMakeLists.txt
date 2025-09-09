get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    MLIRArithDialect
    ${conversion_libs}
    ${dialect_libs}
    MLIROptLib
    MLIRListDialect
    MLIRListToStandard
  )
add_llvm_executable(listproject-opt listproject-opt.cpp)

llvm_update_compile_flags(listproject-opt)
target_link_libraries(listproject-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(listproject-opt)
