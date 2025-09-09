#!/bin/bash

# Script to apply MINIMAL fixes to all exercise branches
# This only fixes build system issues WITHOUT completing any exercises
# All TODO, REMOVE_ME markers are preserved

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Applying minimal build fixes to all branches ===${NC}"

# Save current branch
ORIGINAL_BRANCH=$(git branch --show-current)

# Function to apply minimal CMake fixes to a branch
fix_branch_minimal() {
    local branch=$1
    echo -e "${GREEN}Fixing branch: ${branch}${NC}"
    
    git checkout $branch
    
    # 1. Fix lib/CMakeLists.txt to include Conversion if it exists
    if [ -d "lib/Conversion" ]; then
        if ! grep -q "add_subdirectory(Conversion)" lib/CMakeLists.txt 2>/dev/null; then
            echo -e "${YELLOW}  Adding Conversion subdirectory to lib/CMakeLists.txt${NC}"
            # Add before CAPI if it exists
            if grep -q "add_subdirectory(CAPI)" lib/CMakeLists.txt; then
                sed -i '' '/add_subdirectory(CAPI)/i\
add_subdirectory(Conversion)
' lib/CMakeLists.txt
            else
                echo "add_subdirectory(Conversion)" >> lib/CMakeLists.txt
            fi
        fi
        
        # 2. Create Conversion/CMakeLists.txt if missing
        if [ ! -f "lib/Conversion/CMakeLists.txt" ]; then
            echo -e "${YELLOW}  Creating lib/Conversion/CMakeLists.txt${NC}"
            echo "add_subdirectory(ListToStandard)" > lib/Conversion/CMakeLists.txt
        fi
        
        # 3. Fix ListToStandard/CMakeLists.txt to use add_mlir_library instead of add_mlir_conversion_library
        if [ -f "lib/Conversion/ListToStandard/CMakeLists.txt" ]; then
            if grep -q "add_mlir_conversion_library" lib/Conversion/ListToStandard/CMakeLists.txt; then
                echo -e "${YELLOW}  Fixing lib/Conversion/ListToStandard/CMakeLists.txt${NC}"
                sed -i '' 's/add_mlir_conversion_library/add_mlir_library/g' lib/Conversion/ListToStandard/CMakeLists.txt
                # Remove DEPENDS ListConversionPassIncGen if it exists
                sed -i '' '/DEPENDS.*ListConversionPassIncGen/,+1d' lib/Conversion/ListToStandard/CMakeLists.txt
            fi
        fi
    fi
    
    # 4. Fix listproject-opt/CMakeLists.txt to use proper libraries
    if [ -f "listproject-opt/CMakeLists.txt" ]; then
        echo -e "${YELLOW}  Fixing listproject-opt/CMakeLists.txt${NC}"
        
        # Create a proper CMakeLists.txt that works for all branches
        cat > listproject-opt/CMakeLists.txt << 'EOF'
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
EOF
    fi
    
    # Commit if there are changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        git add -A
        git commit -m "Apply minimal build fixes for CMake compatibility

- Fix CMakeLists.txt files for proper library linking
- Use add_mlir_library instead of add_mlir_conversion_library
- Conditionally include libraries based on availability
- All exercise TODOs and REMOVE_ME markers preserved" || true
    else
        echo -e "${YELLOW}  No changes needed for ${branch}${NC}"
    fi
}

# Apply fixes to all exercise branches
for branch in ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8; do
    if git show-ref --verify --quiet refs/heads/$branch; then
        fix_branch_minimal $branch
    fi
done

# Return to original branch
git checkout $ORIGINAL_BRANCH

echo -e "${GREEN}=== Minimal fixes applied! ===${NC}"
echo -e "${YELLOW}All TODO and REMOVE_ME markers have been preserved for you to complete.${NC}"