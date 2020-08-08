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

// nchw -> nChw16c
struct jit_generic_kernel_t : public jit_generator {
    struct call_params_t {
        const void *in;
        const void *out;
        dims_t pos;
    };

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
        ker_ = (void (*)(const call_params_t *))getCode();
    }

    Address addr_n(int d) {
        return ptr[reg_iptr]; // FIXME
    }

    bool evaluate_predicate(int d) {
        if (prb_.predicates.count(d) == 0) return false;
        const auto &predicate = prb_.predicates.at(d);

        auto reg_acc = reg_pred_evaluation;

        // free dimension
        if (predicate.siblings.size() == 0) {
            cmp(reg_n(d), predicate.restriction);
            return true;
        }

        // connected dimensions
        xor_(reg_acc, reg_acc);
        for (size_t sib = 0; sib < predicate.siblings.size(); ++sib) {
            int n = predicate.siblings[sib];
            if (n < NREGS_N)
                mov(reg_tmp, reg_n(n));
            else
                mov(reg_tmp, addr_n(n));
            if (predicate.factors[sib] != 1)
                imul(reg_tmp, reg_tmp, (int)predicate.factors[sib]);
            add(reg_acc, reg_tmp);
        }
        cmp(reg_acc, predicate.restriction);
        return true;
    }

    void maybe_evaluate_predicate(int d, int unroll, const Label &l_unroll, const Label &l_end) {
        if (prb_.predicates.count(d) == 0) return;
        const auto &predicate = prb_.predicates.at(d);

        auto reg_acc = reg_pred_evaluation;

        cmp(reg_n(d), prb_.nodes[d].n);
        jge(l_end, T_NEAR);

        // free dimension
        if (predicate.siblings.size() == 0)
            return;

        // connected dimensions
        xor_(reg_acc, reg_acc);
        for (size_t _sib = 0; _sib < predicate.siblings.size(); ++_sib) {
            int sib = predicate.siblings.size() - 1 - _sib;
            int n = predicate.siblings[sib];

            if (_sib == 0) {
                int x = utils::div_up(predicate.restriction, predicate.factors[sib]) - 1;
                cmp(reg_n(n), x);
                jl(l_unroll, T_NEAR);
            }

            if (n < NREGS_N)
                mov(reg_tmp, reg_n(n));
            else
                mov(reg_tmp, addr_n(n));
            if (predicate.factors[sib] != 1)
                imul(reg_tmp, reg_tmp, (int)predicate.factors[sib]);
            add(reg_acc, reg_tmp);
        }
        cmp(reg_acc, predicate.restriction - unroll);
        jle(l_unroll, T_NEAR);
        cmp(reg_acc, predicate.restriction);
        jge(l_end, T_NEAR);
    }

    void generate() {
        preamble();

#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_iptr, ptr[reg_param + PARAM_OFF(in)]);
        mov(reg_optr, ptr[reg_param + PARAM_OFF(out)]);
#undef PARAM_OFF

        for (int d = prb_.ndims - 1; d >= 0; --d) xor_(reg_n(d), reg_n(d));
        xor_(reg_ioff, reg_ioff);
        xor_(reg_ooff, reg_ooff);

        std::vector<Label> l_begin(prb_.ndims), l_end(prb_.ndims);

        assert(prb_.ndims <= NREGS_N);
        for (int d = prb_.ndims - 1; d >= 1; --d) {
            L(l_begin[d]);
            if (evaluate_predicate(d)) jge(l_end[d], T_NEAR);
        }

        L(l_begin[0]);
        int unroll = 4;
        Label l_unroll;
        maybe_evaluate_predicate(0, unroll, l_unroll, l_end[0]);

        { // 1
            vmovss(xmm0, ptr[reg_iptr + reg_ioff]);
            vmovss(ptr[reg_optr + reg_ooff], xmm0);

            inc(reg_n(0));
            add(reg_ioff, (int)(prb_.nodes[0].is * sizeof(float)));
            add(reg_ooff, (int)(prb_.nodes[0].os * sizeof(float)));
            jmp(l_begin[0], T_NEAR);
        }

        { // unroll
            L(l_unroll);
            for (int i = 0; i < 4; ++i) {
                Address addr = ptr[reg_iptr + reg_ioff + (int)(i * prb_.nodes[0].is * sizeof(float))];
                pinsrd(xmm0, addr, i);
            }
            movups(ptr[reg_optr + reg_ooff], xmm0);

            add(reg_n(0), unroll);
            add(reg_ioff, (int)(unroll * prb_.nodes[0].is * sizeof(float)));
            add(reg_ooff, (int)(unroll * prb_.nodes[0].os * sizeof(float)));
            jmp(l_begin[0], T_NEAR);
        }

        L(l_end[0]);

        mov(reg_tmp, reg_n(0));
        imul(reg_tmp, reg_tmp, (int)(prb_.nodes[0].is * sizeof(float)));
        sub(reg_ioff, reg_tmp);

        mov(reg_tmp, reg_n(0));
        // imul(reg_tmp, reg_tmp, (int)(prb_.nodes[0].os * sizeof(float)));
        mul_by_const(reg_tmp, reg_tmp2, (int)(prb_.nodes[0].os * sizeof(float)));
        sub(reg_ooff, reg_tmp);

        xor_(reg_n(0), reg_n(0));




        for (int d = 1; d < prb_.ndims; ++d) {
            inc(reg_n(d));
            add(reg_ioff, (int)(prb_.nodes[d].is * sizeof(float)));
            add(reg_ooff, (int)(prb_.nodes[d].os * sizeof(float)));
            jmp(l_begin[d], T_NEAR);

            L(l_end[d]);

            mov(reg_tmp, reg_n(d));
            imul(reg_tmp, reg_tmp, (int)(prb_.nodes[d].is * sizeof(float)));
            sub(reg_ioff, reg_tmp);

            mov(reg_tmp, reg_n(d));
            imul(reg_tmp, reg_tmp, (int)(prb_.nodes[d].os * sizeof(float)));
            sub(reg_ooff, reg_tmp);

            xor_(reg_n(d), reg_n(d));
        }

        postamble();
    }

    void operator()(call_params_t *args) const {
        // printf("ker:%p\n", ker_);
        ker_(args);
    }

private:
    const prb_t &prb_;
    void (*ker_)(const call_params_t *) = nullptr;

    Reg64 reg_tmp = rbx;
    Reg64 reg_tmp2 = r15;
    Reg64 reg_pred_evaluation = rax;

    static constexpr int NREGS_N = 5;
    Reg64 reg_n(int d) {
        assert(d < NREGS_N);
        static Reg64 regs[NREGS_N] = { r8, r9, r10, r11, r14 };
        return regs[d];
    }

    Reg64 reg_param = abi_param1;
    Reg64 reg_iptr = abi_not_param1;
    Reg64 reg_optr = rsi;

    Reg64 reg_ioff = r12;
    Reg64 reg_ooff = r13;
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
            int enable = -1;
            if (enable == -1) {
                const char *env = ::getenv("GR");
                enable = !env || *env == '1';
            }
            if (!enable) return status::unimplemented;

            auto prb = tr::prb_t();

            status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr, true);
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

#if 0
            /* Combine the variables, which appear together on both
             * sides of the reorder */
            prb_simplify(prb);
            DEBUG({
                printf("smpl : ");
                prb_dump(prb);
            });
#endif

            int ndims_par_max = 3;
            tr::prb_t prb_ker;
            std::vector<int> par_dims;
            const auto nthr = dnnl_get_max_threads();
            prb_thread_kernel_balance(prb, ndims_par_max, nthr, prb_ker, par_dims);

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
            _pd->prb_ker_ = prb_ker;
            _pd->par_dims_ = par_dims;
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_, prb_ker_;
        std::vector<int> par_dims_;
    };

    jit_uni_generic_reorder_t(const pd_t *apd) : primitive_t(apd) {
        kernel_ = utils::make_unique<tr::jit_generic_kernel_t>(pd()->prb_ker_);
    }

    void driver_2d(int ithr, int nthr, int off, char *out, const char *in) const {
        for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d1, ptrdiff_t d0) {
                    tr::jit_generic_kernel_t::call_params_t args;
                    args.in = in
                            + (d0 * ns[0].is + d1 * ns[1].is)
                                    * data_type_size(pd()->prb_.itype);
                    args.out = out
                            + (d0 * ns[0].os + d1 * ns[1].os)
                                    * data_type_size(pd()->prb_.otype);

                    (*kernel_)(&args);
                });
    }

    void driver(char *out, const char *in) const {
        const int ndims_ker = pd()->ker_desc_.prb.ndims;

        parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
            driver_2d(ithr, nthr, ndims_ker, out, in);
        });
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
        auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);

        driver(in, out);

        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::jit_generic_kernel_t> kernel_;

    static void prb_thread_kernel_balance(tr::prb_t &prb, int ndims_par_max,
            int nthr, tr::prb_t &prb_ker, std::vector<int> &par_dims) {
        const int ndims = prb.ndims;

        int ndims_par = 0;
        for (int d = prb.ndims - 1; d >= 0 && ndims_par < ndims_par_max; ++d) {
            if (prb.predicates[d].siblings.size() == 1) {
                par_dims.push_back(d);
                ndims_par++;
                printf("par dim: %d\n", d);
            }
        }

        prb_ker = prb;
        for (int n = 0; n < ndims_par; ++n) {
            int d = par_dims[n];
            for (int dd = d; d < ndims - n - 1; ++dd)
                tr::prb_node_swap(prb_ker, dd, dd + 1);
        }
        prb_ker.ndims -= ndims_par;
    }
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
