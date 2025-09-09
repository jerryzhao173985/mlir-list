#!/bin/bash

# Universal CMake build script for MLIR-List tutorial project
# Works across all exercise branches using CMake's native build system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MLIR-List Universal CMake Build Script ===${NC}"

# Set LLVM installation directory
LLVM_INSTALL_DIR="/Users/jerry/llvm-install"
echo -e "${YELLOW}Using LLVM installation at: ${LLVM_INSTALL_DIR}${NC}"

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}Current branch: ${CURRENT_BRANCH}${NC}"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating Python virtual environment...${NC}"
source venv/bin/activate

# Install lit if not already installed
if ! command -v lit &> /dev/null; then
    echo -e "${GREEN}Installing lit...${NC}"
    pip install --upgrade pip
    pip install lit
fi

# Create build directory
echo -e "${GREEN}Creating build directory...${NC}"
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. \
    -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir \
    -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

# Build using cmake --build (works with any generator)
echo -e "${GREEN}Building listproject-opt...${NC}"
cmake --build . --target listproject-opt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Run tests if they exist
echo -e "${GREEN}Checking for tests...${NC}"
if cmake --build . --target help | grep -q "check-listproject"; then
    echo -e "${GREEN}Running tests...${NC}"
    cmake --build . --target check-listproject || {
        echo -e "${YELLOW}⚠ Some tests failed (this is normal for incomplete exercises)${NC}"
    }
else
    echo -e "${YELLOW}No tests found for this branch${NC}"
fi

cd ..

echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}Binary location: ./build/bin/listproject-opt${NC}"
echo -e "${YELLOW}To run the tool: ./build/bin/listproject-opt <input.mlir>${NC}"