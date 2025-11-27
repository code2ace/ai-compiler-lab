#include "qu4d/IR/Qu4dOps.h"
#include "qu4d/IR/Qu4dDialect.h"
using namespace mlir;
using namespace qu4d;

#include "qu4d/IR/Qu4dDialect.cpp.inc"

void Qu4dDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "qu4d/IR/Qu4dOps.cpp.inc"
  >();
}