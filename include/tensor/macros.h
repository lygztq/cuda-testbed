/*!
 * \file macros.h
 * \brief Useful macros used in tensor
 */
#ifndef TENSOR_MACROS_H_
#define TENSOR_MACROS_H_

#ifndef TENSOR_DLL
#ifdef _WIN32
#ifdef TENSOR_EXPORTS
#define TENSOR_DLL __declspec(dllexport)
#else // TENSOR_EXPORTS
#define TENSOR_DLL __declspec(dllimport)
#endif // TENSOR_EXPORTS
#else // _WIN32
#define TENSOR_DLL
#endif // _WIN32
#endif // TENSOR_DLL

#define RESTRICT __restrict

#define TENSOR_KERNEL __global__
#define TENSOR_HOST   __host__
#define TENSOR_DEVICE __device__
#define TENSOR_HOST_DEVICE TENSOR_HOST TENSOR_DEVICE
#define CUDA_LAMBDA TENSOR_DEVICE TENSOR_HOST

#endif // TENSOR_MACROS_H_
