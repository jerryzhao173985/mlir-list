get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${conversion_libs}
    ${dialect_libs}
    MLIROptLib
    MLIRListDialect
  )

# Add MLIRListToStandard if it exists
if(TARGET MLIRListToStandard)
    list(APPEND LIBS MLIRListToStandard)
endif()

# Add MLIRListTransforms if it exists  
if(TARGET MLIRListTransforms)
    list(APPEND LIBS MLIRListTransforms)
endif()

add_llvm_executable(listproject-opt listproject-opt.cpp)

llvm_update_compile_flags(listproject-opt)
target_link_libraries(listproject-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(listproject-opt)
