#!/bin/bash

# Build script for MLIR-List tutorial project using local LLVM installation
# This script is designed to work across all exercise branches

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MLIR-List Universal Build Script ===${NC}"

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
    -G Ninja \
    -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir \
    -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

# Build
echo -e "${GREEN}Building listproject-opt...${NC}"
ninja listproject-opt

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
ninja check-listproject

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    # Don't exit with error for failing tests during exercises
fi

cd ..

echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}To use the tools, run: source tosource.sh${NC}"
echo -e "${YELLOW}Binary location: ./build/bin/listproject-opt${NC}"
