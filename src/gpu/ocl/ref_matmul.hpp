/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_REF_MATMUL_HPP
#define GPU_OCL_REF_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_matmul_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;
            bia_dt_ = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool ok = IMPLICATION(desc()->accum_data_type == s32,
                              attr()->zero_points_.common())
                    && IMPLICATION(desc()->accum_data_type != s32,
                            attr()->zero_points_.has_default_values())
                    && attr()->has_default_values(smask_t::oscale_runtime
                            | smask_t::zero_points_runtime | smask_t::post_ops)
                    && attr_oscale_ok() && set_default_formats()
                    && !has_blocks()
                    && ((utils::one_of(src_dt_, u8, s8)
                                && utils::one_of(wei_dt_, u8, s8)
                                && utils::one_of(dst_dt_, f32, s8, u8, s32, f16)
                                && IMPLICATION(with_bias(),
                                        utils::one_of(
                                                bia_dt_, f32, u8, s8, s32)))
                            || ((utils::everyone_is(
                                         f32, src_dt_, wei_dt_, dst_dt_)
                                        || (utils::everyone_is(
                                                    f16, src_dt_, wei_dt_)
                                                && utils::one_of(
                                                        dst_dt_, u8, s8, f16))
                                        || (utils::everyone_is(
                                                    bf16, src_dt_, wei_dt_)
                                                && utils::one_of(
                                                        dst_dt_, bf16, f32)))
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt_, f32))))
                    && post_ops_with_binary_ok(attr(), dst_dt_, 6)
                    && attr_.set_default_formats(dst_md(0)) == status::success;

            if (!ok) return status::unimplemented;

            non_default_attrs_ = !attr()->has_default_values();
            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        bool non_default_attrs_ = false;
        data_type_t bia_dt_ = data_type::undef;
        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;

        attr_info_t attr_info_ = {};

    private:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0 || oscale.mask_ == (1 << (batched() + 1));
        }
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("DST_NDIMS", pd()->dst_md()->ndims);
        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int("NON_DEFAULT_ATTRS", pd()->non_default_attrs_);

        kernel_ctx.set_data_type(pd()->dst_dt_);
        def_attr_info(kernel_ctx, pd()->attr_info_, pd()->attr()->post_ops_);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->bia_dt_, "BIA");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");
        create_kernel(engine, &kernel_, "ref_matmul", kernel_ctx);
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif