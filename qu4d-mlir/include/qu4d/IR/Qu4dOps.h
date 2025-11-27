#ifndef QU4D_IR_QU4DOPS_H
#define QU4D_IR_QU4DOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect header (declares mlir::qu4d::Qu4dDialect)
#include "qu4d/IR/Qu4dDialect.h"

// Generated op decls
//#include "qu4d/IR/Qu4dDialect.h.inc"
#define GET_OP_CLASSES
#include "qu4d/IR/Qu4dOps.h.inc"

#endif // QU4D_IR_QU4DOPS_H
