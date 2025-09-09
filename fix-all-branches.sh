#!/bin/bash

# Script to fix all exercise branches to work with the local LLVM installation
# This ensures each branch has the proper CMake configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Fixing all MLIR-List exercise branches ===${NC}"

# Save current branch
ORIGINAL_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}Current branch: ${ORIGINAL_BRANCH}${NC}"

# Function to fix a branch
fix_branch() {
    local branch=$1
    echo -e "${GREEN}Fixing branch: ${branch}${NC}"
    
    git checkout $branch
    
    # Ensure lib/CMakeLists.txt includes Conversion if it exists
    if [ -d "lib/Conversion" ]; then
        if ! grep -q "add_subdirectory(Conversion)" lib/CMakeLists.txt 2>/dev/null; then
            echo -e "${YELLOW}  Adding Conversion subdirectory to lib/CMakeLists.txt${NC}"
            # Add before CAPI if it exists, otherwise at the end
            if grep -q "add_subdirectory(CAPI)" lib/CMakeLists.txt; then
                sed -i '' '/add_subdirectory(CAPI)/i\
add_subdirectory(Conversion)
' lib/CMakeLists.txt
            else
                echo "add_subdirectory(Conversion)" >> lib/CMakeLists.txt
            fi
        fi
        
        # Fix the ListToStandard CMakeLists.txt if it exists
        if [ -f "lib/Conversion/ListToStandard/CMakeLists.txt" ]; then
            echo -e "${YELLOW}  Fixing lib/Conversion/ListToStandard/CMakeLists.txt${NC}"
            
            # Check which pattern the file uses and fix accordingly
            if grep -q "ListConversionPassIncGen" lib/Conversion/ListToStandard/CMakeLists.txt 2>/dev/null; then
                # Remove the invalid DEPENDS line
                sed -i '' '/DEPENDS.*ListConversionPassIncGen/d' lib/Conversion/ListToStandard/CMakeLists.txt
            fi
        fi
    fi
    
    # Fix listproject-opt/CMakeLists.txt to conditionally link MLIRListToStandard
    if [ -f "listproject-opt/CMakeLists.txt" ]; then
        echo -e "${YELLOW}  Fixing listproject-opt/CMakeLists.txt${NC}"
        
        # Check if MLIRListToStandard should be available
        if [ -d "lib/Conversion/ListToStandard" ]; then
            # Ensure MLIRListToStandard is in LIBS if Conversion exists
            if ! grep -q "MLIRListToStandard" listproject-opt/CMakeLists.txt; then
                sed -i '' '/set(LIBS/,/)/s/)$/\n    MLIRListToStandard\n  )/' listproject-opt/CMakeLists.txt
            fi
        else
            # Remove MLIRListToStandard if Conversion doesn't exist
            sed -i '' '/MLIRListToStandard/d' listproject-opt/CMakeLists.txt
        fi
        
        # Ensure MLIRArithDialect is included only once and in the right place
        # Remove any existing MLIRArithDialect entries
        sed -i '' '/MLIRArithDialect/d' listproject-opt/CMakeLists.txt
        # Add it back properly
        sed -i '' '/set(LIBS/a\
    MLIRArithDialect
' listproject-opt/CMakeLists.txt
    fi
    
    # Copy the universal build script to this branch
    if [ -f "../build-cmake.sh" ]; then
        cp ../build-cmake.sh build-cmake.sh
        chmod +x build-cmake.sh
    fi
    
    # Commit the fixes
    if git diff --quiet && git diff --cached --quiet; then
        echo -e "${YELLOW}  No changes needed for ${branch}${NC}"
    else
        git add -A
        git commit -m "Fix CMake configuration for local LLVM build

- Ensure Conversion subdirectory is included when present
- Fix ListToStandard library dependencies
- Conditionally link MLIRListToStandard based on availability
- Add universal CMake build script" || true
    fi
    
    echo -e "${GREEN}  ✓ Fixed ${branch}${NC}"
}

# Fix all exercise branches
for branch in ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8; do
    if git show-ref --verify --quiet refs/heads/$branch; then
        fix_branch $branch
    else
        echo -e "${RED}Branch ${branch} does not exist, skipping${NC}"
    fi
done

# Also fix solution branches if they exist
for branch in ex1-solution ex2-solution ex3-solution ex4-solution ex5-solution ex6-solution ex7-solution ex8-solution; do
    if git show-ref --verify --quiet refs/heads/$branch; then
        fix_branch $branch
    else
        echo -e "${YELLOW}Solution branch ${branch} does not exist, skipping${NC}"
    fi
done

# Return to original branch
echo -e "${GREEN}Returning to original branch: ${ORIGINAL_BRANCH}${NC}"
git checkout $ORIGINAL_BRANCH

echo -e "${GREEN}=== All branches fixed! ===${NC}"
echo -e "${YELLOW}You can now switch to any exercise branch and run:${NC}"
echo -e "${YELLOW}  ./build-cmake.sh${NC}"