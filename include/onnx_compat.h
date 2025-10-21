#ifndef ONNX_COMPAT_H_
#define ONNX_COMPAT_H_

#include <cstdint>

// Platform-specific compatibility fixes for ONNX Runtime

// Fix for platforms without proper __fp16 support (especially RISC-V)
#if defined(__riscv) || defined(__riscv__)
  // RISC-V specific fixes
  #ifndef __fp16
    #ifdef __riscv_f
      // If floating point is supported, try to use _Float16
      #ifdef __FLT16_MAX__
        typedef _Float16 __fp16;
      #else
        // Fallback to software emulation
        typedef struct { uint16_t __v; } __fp16;
      #endif
    #else
      // No hardware FP support, use software emulation
      typedef struct { uint16_t __v; } __fp16;
    #endif
  #endif
#elif defined(__arm__) || defined(__aarch64__)
  // ARM specific fixes
  #ifndef __fp16
    #if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE)
      // ARM has native __fp16 support, should be defined already
    #else
      // Fallback for older ARM without fp16
      typedef struct { uint16_t __v; } __fp16;
    #endif
  #endif
#else
  // Generic fallback for other platforms
  #ifndef __fp16
    #ifdef _Float16
      typedef _Float16 __fp16;
    #else
      typedef struct { uint16_t __v; } __fp16;
    #endif
  #endif
#endif

// Workaround for missing union member 'f' in FP16_Union_Bits
// This is a hack that might be needed for some ONNX Runtime versions
#ifdef __riscv
  #define ONNXRUNTIME_FLOAT16_WORKAROUND
#endif

// Include ONNX Runtime headers after compatibility fixes
#include <onnxruntime_cxx_api.h>

// Additional workarounds if needed
#ifdef ONNXRUNTIME_FLOAT16_WORKAROUND
  // If the union issue persists, we might need to patch it here
  // But this requires knowing the exact structure
#endif

#endif // ONNX_COMPAT_H_
