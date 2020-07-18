/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include <assert.h>
#include <numeric>

#include "dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/cpu_reorder_pd.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) \
    do { \
        __VA_ARGS__ \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
constexpr static bool is_windows = true;
#else
constexpr static bool is_windows = false;
#endif

using namespace Xbyak;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

struct jit_generic_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_generic_kernel_t)

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims >= 2 && mayiuse(avx2)
                && p.scale_type == scale_type_t::NONE
                && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
                && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f;
        if (!ok) return false;

        return ok;
    }

    jit_generic_kernel_t(const tr::prb_t &prb)
        : jit_generator(), prb_(prb) {
        generate();
        ker_ = (void (*)(void *, void *))getCode();
    }

    Address addr_n(int d) {
        return ptr[reg_ptr_in + reg_off_in + i_off * itype_sz];
    }

    bool evaluate_predicate(int d) {
        if (prb_.predicates.count(d) == 0) return false;
        const auto &predicate = prb_.predicates[d];

        auto reg_tmp = reg_pred_evaluation[0];
        assert(reg_tmp == rax); // mul
        auto reg_acc = reg_pred_evaluation[1];

        // free dimension
        if (predicate.siblings.size() == 0) {
            cmp(reg_n(d), predicate.restriction);
            return true;
        }

        // connected dimension
        xor_(reg_acc, reg_acc, reg_acc);
        for (size_t sib = 0; sib < predicate.siblings.size(); ++sib) {
            int n = predicate.siblings[sib];
            if (n < NREGS_N)
                mov(reg_tmp, reg_n(n));
            else
                mov(reg_tmp, addr_n(n));
            mul(predicate.factors[sib]);
            add(reg_acc, reg_acc, reg_tmp);
        }
        cmp(reg_acc, predicate.restriction);
        return true;
    }

    void generate() {
        preamble();

        xor_(reg_ioff, reg_ioff, reg_ioff);
        xor_(reg_ooff, reg_ooff, reg_ooff);

        std::vector<Label> l_begin(prb_.ndims), l_end(prb_.ndims);

        assert(prb_.ndims <= NREGS_N);
        for (int d = 0; d < prb_.ndims; ++d) {
            xor_(reg_n(d), reg_n(d), reg_n(d));
            L(l_begin(d));
            if (evaluate_predicate(d))
                jge(l_end(d));
        }

        vmovss(xmm0, ptr[reg_iptr + reg_ioff]);
        vmovss(ptr[reg_optr + reg_ooff], xmm0);

        for (int d = prb_.ndims - 1; d >= 0; --d) {
            inc(reg_n(d));
            jmp(l_begin(d));

            L(l_end(d));

            mov(reg_tmp, reg_n(d));
            imul(reg_tmp, prb_.nodes[d].is * sizeof(float));
            sub(reg_ioff, reg_tmp);

            mov(reg_tmp, reg_n(d));
            imul(reg_tmp, prb_.nodes[d].os * sizeof(float));
            sub(reg_ooff, reg_tmp);
        }

        postamble();
    }

    void operator()(const void *in, void *out) const { ker_(in, out); }

private:
    const prb_t &prb_;
    void (*ker_)(const void *in, void *out) = nullptr;

    Reg64 reg_tmp = rbx;
    static Reg64 reg_pred_evaluation[2] = { rax, rbx };

    constexpr int NREGS_N = 4;
    Reg64 reg_n(int d) {
        static Reg64 regs[NREGS_N] = { r8, r9, r10, r11 };
        return regs[d];
    }

    Reg64 reg_iptr = rdi;
    Reg64 reg_optr = rsi;
};

} // namespace tr

struct jit_uni_generic_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;
        DECLARE_COMMON_PD_T("jit:uni:generic", jit_uni_generic_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
            if (prb_init_status != status::success) return prb_init_status;

            DEBUG({
                printf("init : ");
                prb_dump(prb);
            });
            // Sort the prb array in increasing sizes of the output stride
            prb_normalize(prb);
            DEBUG({
                printf("norm : ");
                prb_dump(prb);
            });
            /* Combine the variables, which appear together on both
             * sides of the reorder */
            prb_simplify(prb);
            DEBUG({
                printf("smpl : ");
                prb_dump(prb);
            });
            DEBUG({
                printf("tile : ");
                prb_dump(prb);
            });

            if (!tr::jit_generic_kernel_t::applicable(prb)) {
                return status::unimplemented;
            }

            auto _pd = new pd_t(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            _pd->prb_ = prb;
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
    };

    jit_uni_generic_reorder_t(const pd_t *apd) : primitive_t(apd) {
        kernel_ = utils::make_unique<tr::jit_generic_kernel_t>(pd()->prb_);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
        auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::jit_generic_kernel_t> kernel_;
};

status_t jit_uni_generic_reorder_create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    return jit_uni_generic_reorder_t::pd_t::create(reorder_pd, engine, attr,
                src_engine, src_md, dst_engine, dst_md);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
