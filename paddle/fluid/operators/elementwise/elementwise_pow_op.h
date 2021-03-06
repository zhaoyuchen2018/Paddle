/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cmath>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct PowFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return std::pow(a, b); }
};

template <typename DeviceContext, typename T>
class ElementwisePowKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::LoDTensor;
    auto* x = ctx.Input<Tensor>("X");
    PADDLE_ENFORCE(x != nullptr,
                   "Cannot get input Variable X, variable name = %s",
                   ctx.op().Input("X"));
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<PowFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          PowFunctor<T>(), z);
  }
};

template <typename T>
struct PowGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * y * std::pow(x, y - 1);
  }
};

template <typename T>
struct PowGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * std::log(x) * std::pow(x, y);
  }
};

template <typename DeviceContext, typename T>
class ElementwisePowGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, PowGradDX<T>, PowGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, PowGradDX<T>(), PowGradDY<T>());
  }
};
}  // namespace operators
}  // namespace paddle
