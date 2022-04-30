#include "tensor/iterator.h"

namespace tensor {

void TensorIterator::Build() {
  FixTensors();
  InitShape();
  BroadcastShape();
  CompressShape();
}

void TensorIterator::FixTensors() {
  num_in_tensors_ = in_tensors_.size();
  num_out_tensors_ = out_tensors_.size();

  tensors_ = std::move(out_tensors_);
  tensors_.resize(num_in_tensors_ + num_out_tensors_);
  std::copy_n(in_tensors_.begin(), num_in_tensors_, tensors_.begin() + num_out_tensors_);
  in_tensors_.clear();
  has_fixed_tensor_ = true;
}

void TensorIterator::InitShape() {
  CHECK(has_fixed_tensor_);

  size_t max_axis = 0;
  for (const auto& v : tensors_)
    max_axis = std::max(max_axis, v.NumAxes());

  shape_.resize(max_axis);
  for (auto& s : shape_) s = 1;

  has_init_ = true;
}

void TensorIterator::BroadcastShape() {
  CHECK(has_init_);

  for (auto& tensor : tensors_) {
    TensorShapeInfo& shape_info = tensor.GetShapeInfo();
    auto& tensor_shape = shape_info.ShapeRef();
    auto& tensor_stride = shape_info.StrideRef();
    size_t offset = NumAxes() - tensor.NumAxes();
    shape_info.ChangeNumAxis(NumAxes());

    for (int i = static_cast<int>(NumAxes() - 1); i >= static_cast<int>(offset); --i) {
      if (tensor_shape[i - offset] == 1) {
        tensor_stride[i] = 0;
      } else {
        if (shape_[i] == 1) {
          shape_[i] = tensor_shape[i - offset];
        } else if (shape_[i] != tensor_shape[i - offset]) {
          LOG_ERROR << "Tensor shape (" << tensor_shape[i - offset]
                    <<") does not match shape(" << shape_[i]
                    <<") at axis (" << i <<").\n";
        }
        tensor_stride[i] = tensor_stride[i - offset];
      }
      tensor_shape[i] = tensor_shape[i - offset];
    }

    std::for_each_n(tensor_shape.begin(), offset, [](auto& n) { n = 1; });
    std::for_each_n(tensor_stride.begin(), offset, [](auto& n) { n = 0; });
  }

  has_broadcasted_shape_ = true;
}

// Compress local contiguous shape/stride axis
// interval for better performance
void TensorIterator::CompressShape() {
  // no need for this
  if (NumAxes() <= 1) return;

  auto CanCompress = [&](size_t dim0, size_t dim1) -> bool {
    if (dim0 >= dim1) return false;
    if (shape_[dim0] == 1 || shape_[dim1] == 1) return true;
    for (auto& tensor : tensors_) {
      TensorShapeInfo& shape_info = tensor.GetShapeInfo();
      auto& tensor_shape = shape_info.ShapeRef();
      auto& tensor_stride = shape_info.StrideRef();
      if ((tensor_stride[dim0] != tensor_stride[dim1] * tensor_shape[dim1]) &&
          (tensor_stride[dim0] * tensor_stride[dim1] != 0)) {
            return false;
      }
    }
    return true;
  };

  auto Compress = [&](size_t dim0, size_t dim1) {
    for (auto& tensor : tensors_) {
      TensorShapeInfo& shape_info = tensor.GetShapeInfo();
      auto& tensor_shape = shape_info.ShapeRef();
      auto& tensor_stride = shape_info.StrideRef();
      if (tensor_shape[dim1] == 1) {
        tensor_shape[dim1] = tensor_shape[dim0];
        tensor_stride[dim1] = tensor_shape[dim0];
      } else if (tensor_shape[dim0] != 1) {
        tensor_shape[dim1] *= tensor_shape[dim0];
      }
      // set dim0 invalid (has been eaten)
      tensor_shape[dim0] = 1;
      tensor_stride[dim0] = 0;
    }
    shape_[dim1] *= shape_[dim0];
  };

  auto RemapArray = [&](size_t offset) {
    if (!offset) return;
    for (auto& tensor : tensors_) {
      TensorShapeInfo& shape_info = tensor.GetShapeInfo();
      auto& tensor_shape = shape_info.ShapeRef();
      auto& tensor_stride = shape_info.StrideRef();
      for (size_t i = 0; i < shape_info.NumAxes() - offset; ++i) {
        tensor_shape[i] = tensor_shape[i + offset];
        tensor_stride[i] = tensor_stride[i + offset];
      }
      shape_info.ChangeNumAxis(shape_info.NumAxes() - offset);
    }
    for (size_t i = 0; i < NumAxes(); ++i) {
      shape_[i] = shape_[i + offset];
    }
    shape_.resize(NumAxes() - offset);
  };

  size_t num_compress = 0;
  size_t prev_dim = NumAxes() - 1;
  size_t start_dim = prev_dim - 1;
  for (size_t next_dim = start_dim; next_dim <= start_dim; --next_dim) {
    if (CanCompress(next_dim, prev_dim)) {
      Compress(next_dim, prev_dim);
      ++num_compress;
    } else {
      --prev_dim;
      if (prev_dim != next_dim) {
        Compress(next_dim, prev_dim);
      }
    }
  }

  RemapArray(num_compress);
  has_compressed_ = true;
}

std::vector<size_t> TensorIterator::getStridesInBytes() const {
  size_t num_tensors = NumTensors();
  size_t num_axes = NumAxes();
  std::vector<size_t> strides(num_tensors * num_axes, 0);

  for (size_t t = 0; t < tensors_.size(); ++t) {
    for (size_t i = 0; i < num_axes; ++i) {
      strides[i + t * num_axes] = tensors_[t].Stride(i) * tensors_[t].ElemSize();
    }
  }

  return strides;
}

void TensorIterator::getDataPtrs(std::vector<char*>& dptrs,
                                 const std::vector<char*>& base,
                                 const std::vector<size_t>& index,
                                 const std::vector<size_t>& stride_bytes) const {
  size_t num_tensors = NumTensors();
  size_t num_axes = NumAxes();
  CHECK_EQ(dptrs.size(), num_tensors);
  CHECK_EQ(index.size(), num_axes);

  for (size_t t = 0; t < num_tensors; ++t) {
     size_t offset = 0;
     for (size_t i = 0; i < num_axes; ++i) {
       offset += index[i] * stride_bytes[t * num_axes + i];
     }
     dptrs[t] = base[t] + offset;
  }
}

void TensorIterator::ForEach(loop2d_t loop) {
  CHECK(Valid());

  // init data ptrs
  size_t num_tensors = NumTensors();
  size_t num_axes = NumAxes();
  std::vector<char*> base_dptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    base_dptrs[i] = reinterpret_cast<char*>(tensors_[i].RawPtr());
  }

  auto elem_to_bytes = [&](size_t elem, size_t alignment) {
    return elem * alignment;
  };

  // init stride for loop
  size_t stride_axis = std::max(num_axes, 2ULL);
  int stride_idx = static_cast<int>(stride_axis - 1);
  std::vector<size_t> loop_stride(num_tensors * stride_axis, 0);
  for (int i = static_cast<int>(num_axes - 1); i >= 0; --i, --stride_idx) {
    for (size_t t = 0; t < tensors_.size(); ++t) {
      loop_stride[stride_idx * num_tensors + t] = elem_to_bytes(
        tensors_[t].Stride(i), tensors_[t].ElemSize());
    }
  }
  size_t inner_size = shape_[num_axes - 1];
  size_t outer_size = (num_axes > 1) ? shape_[num_axes - 2] : 1;

  if (num_axes <= 2) {
    loop(base_dptrs.data(), loop_stride.data(), inner_size, outer_size);
  } else {
    auto counter = IndexCounter(shape_);
    std::vector<char*> dptrs(num_tensors);
    std::vector<size_t> stride_bytes = getStridesInBytes();
    while (!counter.IsFinish()) {
      getDataPtrs(dptrs, base_dptrs, counter.Index(), stride_bytes);
      loop(dptrs.data(), loop_stride.data(), inner_size, outer_size);
      counter.Advance(num_axes - 3);
    }
  }
}

} // namespace tensor
