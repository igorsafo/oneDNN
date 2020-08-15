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
#include <algorithm>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "dnnl_debug.h"

#include "cpu/x64/jit_uni_reorder.hpp"

using namespace dnnl::impl::types;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

/** ad-hoc structure to describe blocked memory layout */
struct layout_desc_t {
    data_type_t dt;
    int ndims;
    dims_t id;
    dims_t dims;
    strides_t strides;
};

status_t cvt_mem_desc_to_layout_desc(
        const memory_desc_t &md_, layout_desc_t &ld, const dims_t &blocks) {
    const auto md = memory_desc_wrapper(md_);

    bool ok = true && md.is_blocking_desc() && md.extra().flags == 0;
    if (!ok) return invalid_arguments;

    const auto &bd = md.blocking_desc();

    ld.ndims = 0;
    ld.dt = md.data_type();

    auto P = [&ld](int id, int dim, ptrdiff_t stride) {
        assert((size_t)ld.ndims < sizeof(ld.dims) / sizeof(ld.dims[0]));
        ld.id[ld.ndims] = id;
        ld.dims[ld.ndims] = dim;
        ld.strides[ld.ndims] = stride;
        ++ld.ndims;
    };

    for (int d = 0; d < md.ndims(); ++d) {
        const int ld_ndims_start = ld.ndims;
        if (blocks[d] != 1) {
            stride_t stride = 1;
            for (int iblk = bd.inner_nblks - 1; iblk >= 0; --iblk) {
                if (bd.inner_idxs[iblk] == d) P(d, bd.inner_blks[iblk], stride);
                stride *= bd.inner_blks[iblk];
            }
        }
        P(d, md.padded_dims()[d] / blocks[d], bd.strides[d]);

        // TODO: NOW: revisit, do we need a reverse?
        // TODO: NOW: consider using strides instead of block sizes in md
        // reverse the order of dims
        for (int ld_d = 0; ld_d < (ld.ndims - ld_ndims_start) / 2; ++ld_d) {
            const int idx0 = ld_ndims_start + ld_d;
            const int idx1 = ld.ndims - 1 - ld_d;
            nstl::swap(ld.dims[idx0], ld.dims[idx1]);
            nstl::swap(ld.strides[idx0], ld.strides[idx1]);
        }
    }

    return success;
}

status_t prb_init(prb_t &p, const memory_desc_t &imd_,
        const memory_desc_t &omd_, const primitive_attr_t *attr,
        bool use_predicates) {
    memory_desc_t imd = imd_;
    memory_desc_t omd = omd_;
    auto im_d = memory_desc_wrapper(imd);
    auto om_d = memory_desc_wrapper(omd);

    auto check_post_ops = [](const primitive_attr_t *attr) {
        const auto &po = attr->post_ops_;
        return po.len() == 0
                || (po.len() == 1 && po.contain(primitive_kind::sum, 0));
    };

    bool ok = im_d.is_blocking_desc() && om_d.is_blocking_desc()
            && !im_d.has_runtime_dims_or_strides() && !im_d.has_zero_dim()
            && !om_d.has_runtime_dims_or_strides() && !om_d.has_zero_dim()
            && attr->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_post_ops(attr);
    if (!ok) return unimplemented;

    dims_t iblocks, oblocks;
    im_d.compute_blocks(iblocks);
    om_d.compute_blocks(oblocks);

    enum class changed_md_t { none, input, output };
    changed_md_t changed_md = changed_md_t::none;
    uint8_t changed_dims[DNNL_MAX_NDIMS] = { 0 }; // 1 if changed

    /* padding_dim consistency check */
    for (int d = 0; d < im_d.ndims(); ++d) {
        // dirty hack to override the padding and simplify the logic below
        // FIXME: doesn't work if changed dim has blocking
        if (im_d.padded_dims()[d] < om_d.padded_dims()[d]) {
            if (!use_predicates) return unimplemented;
            auto want_to_change = changed_md_t::input;
            if (changed_md != want_to_change) {
                if (changed_md != changed_md_t::none) return unimplemented;
                changed_md = want_to_change;
            }
            changed_dims[d] = 1;
            imd.padded_dims[d] = om_d.padded_dims()[d];
        } else if (im_d.padded_dims()[d] > om_d.padded_dims()[d]) {
            if (!use_predicates) return unimplemented;
            auto want_to_change = changed_md_t::output;
            if (changed_md != want_to_change) {
                if (changed_md != changed_md_t::none) return unimplemented;
                changed_md = want_to_change;
            }
            changed_dims[d] = 1;
            omd.padded_dims[d] = im_d.padded_dims()[d];
        }

        auto pdim = imd.padded_dims[d];
        bool ok = pdim % iblocks[d] == 0 && pdim % oblocks[d] == 0;
        if (!ok) return unimplemented;
    }

    layout_desc_t ild, old;
    status_t status = cvt_mem_desc_to_layout_desc(imd, ild, iblocks);
    if (status != success) return status;
    status = cvt_mem_desc_to_layout_desc(omd, old, oblocks);
    if (status != success) return status;

    p.itype = ild.dt;
    p.otype = old.dt;

    p.scale_type = attr->output_scales_.has_default_values()
            ? scale_type_t::NONE
            : (attr->output_scales_.mask_ == 0 ? scale_type_t::COMMON
                                               : scale_type_t::MANY);

    ptrdiff_t ss[max_ndims] = {0};
    if (p.scale_type == scale_type_t::MANY) {
        ptrdiff_t last_ss = 1;
        for (int d = old.ndims - 1; d >= 0; --d) {
            assert((d == 0 || old.id[d - 1] <= old.id[d])
                    && "logical dimensions should be in ascending order");
            if (attr->output_scales_.mask_ & (1 << old.id[d])) {
                ss[d] = last_ss;
                last_ss *= old.dims[d];
            }
        }
    }

    int ndims = 0;

    int i_pos = 0; /* state for input  -- current dimension */
    int o_pos = 0; /* state for output -- current dimension */

    int p_ids[DNNL_MAX_NDIMS];

    while (i_pos < ild.ndims && o_pos < old.ndims) {
        assert(ild.id[i_pos] == old.id[o_pos]);
        if (ild.id[i_pos] != old.id[o_pos]) return runtime_error;

        assert(ndims < max_ndims);
        if (ndims == max_ndims) return runtime_error;

        p_ids[ndims] = ild.id[i_pos];

        if (ild.dims[i_pos] == old.dims[o_pos]) {
            p.nodes[ndims].n = ild.dims[i_pos];
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            ++ndims;
            ++i_pos;
            ++o_pos;
        } else if (ild.dims[i_pos] < old.dims[o_pos]) {
            assert(old.dims[o_pos] % ild.dims[i_pos] == 0);
            int factor = old.dims[o_pos] / ild.dims[i_pos];
            p.nodes[ndims].n = ild.dims[i_pos];
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos] * factor;
            p.nodes[ndims].ss = ss[o_pos] * factor;
            ++ndims;
            ++i_pos;
            old.dims[o_pos] = factor;
        } else if (ild.dims[i_pos] > old.dims[o_pos]) {
            assert(ild.dims[i_pos] % old.dims[o_pos] == 0);
            int factor = ild.dims[i_pos] / old.dims[o_pos];
            p.nodes[ndims].n = old.dims[o_pos];
            p.nodes[ndims].is = ild.strides[i_pos] * factor;
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            ++ndims;
            ++o_pos;
            ild.dims[i_pos] = factor;
        }
    }
    p.ndims = ndims;

    for (int d = p.ndims - 1; d >= 0; ) {
        int dim_id = p_ids[d];
        if (!changed_dims[dim_id]) {
            p.predicates[d] = {{}, {}, p.nodes[d].n};
            d -= 1;
            continue;
        }

        std::vector<int> siblings;
        std::vector<int> factors;
        size_t restriction = im_d.dims()[dim_id];

        int nsibs = 0;
        int factor = 1;
        while (d - nsibs >= 0 && p_ids[d - nsibs] == p_ids[d]) {
            int sib = d - nsibs;
            siblings.push_back(sib);
            factors.push_back(factor);

            factor *= p.nodes[d - nsibs].n;
            nsibs += 1;
        }

        for (int isib = 0; isib < nsibs; ++isib)
            p.predicates[d - isib] = {siblings, factors, restriction};

        d -= nsibs;
    }

    p.ioff = memory_desc_wrapper(imd).offset0();
    p.ooff = memory_desc_wrapper(omd).offset0();

    const int sum_idx = attr->post_ops_.find(primitive_kind::sum);
    p.beta = sum_idx == -1 ? 0.f : attr->post_ops_.entry_[sum_idx].sum.scale;

    return success;
}

static bool prb_check_free_dim(const prb_t &p, int dim) {
    return p.predicates.empty() || p.predicates.at(dim).siblings.size() == 0;
}

void prb_normalize(prb_t &p) {
    for (int d = 0; d < p.ndims; ++d) {
        int min_pos = d;
        for (int j = d + 1; j < p.ndims; ++j) {
            bool new_min = p.nodes[j].os < p.nodes[min_pos].os
                    || (p.nodes[j].os == p.nodes[min_pos].os
                            && p.nodes[j].n < p.nodes[min_pos].n);
            if (new_min) min_pos = j;
        }
        if (min_pos != d) prb_node_swap(p, d, min_pos);
    }
}

void prb_simplify(prb_t &p) {
#if defined(__GNUC__) && __GNUC__ >= 4
/* GCC produces bogus array subscript is above array bounds warning for
 * the `p.nodes[j - 1] = p.nodes[j]` line below, so disable it for now. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    for (int d = 0; d < p.ndims - 1; ++d) {
        if (!prb_check_free_dim(p, d) || !prb_check_free_dim(p, d + 1))
            continue;

        auto &this_node = p.nodes[d + 0];
        auto &next_node = p.nodes[d + 1];
        const bool fold = false
                || next_node.n == (size_t)1 // trivial case, just drop next node
                || (true // or real folding if possible
                        && next_node.is == (ptrdiff_t)this_node.n * this_node.is
                        && next_node.os == (ptrdiff_t)this_node.n * this_node.os
                        && next_node.ss
                                == (ptrdiff_t)this_node.n * this_node.ss);
        if (fold) {
            this_node.n *= next_node.n;
            for (int j = d + 2; j < p.ndims; ++j)
                p.nodes[j - 1] = p.nodes[j];
            --p.ndims;
            --d; // make another try
        }
    }
#if defined(__GNUC__) && __GNUC__ >= 4
#pragma GCC diagnostic pop
#endif
}

void prb_node_split(prb_t &p, int dim, size_t n1) {
    assert(dim < p.ndims);
    assert(p.ndims < max_ndims);
    assert(p.nodes[dim].n % n1 == 0);

    // two step implementation: add split dim to the end, and then move
    // TODO: optimize

    p.ndims += 1;

    // step 1: split dim into (dim) and (ndims-1)

    p.nodes[p.ndims - 1].n = p.nodes[dim].n / n1;
    p.nodes[p.ndims - 1].is = p.nodes[dim].is * n1;
    p.nodes[p.ndims - 1].os = p.nodes[dim].os * n1;
    p.nodes[p.ndims - 1].ss = p.nodes[dim].ss * n1;

    p.nodes[dim].n = n1;

    if (prb_check_free_dim(p, dim)) {
        if (!p.predicates.empty()) {
            // simple case: independent dim
            p.predicates[dim - 1].restriction = p.nodes[dim].n;
            p.predicates[p.ndims - 1] = {{}, {}, p.nodes[p.ndims - 1].n};
        }
    } else {
        std::vector<int> &siblings = p.predicates[dim].siblings;
        std::vector<int> &factors = p.predicates[dim].factors;

        size_t sib_dim = 0;
        for (; sib_dim < siblings.size(); ++sib_dim)
            if (siblings[sib_dim] == dim) break;

        siblings.push_back(p.ndims - 1);
        factors.push_back(factors[sib_dim] * p.nodes[dim].n);

        for (int sib: siblings) {
            if (sib == dim || sib == p.ndims - 1) continue;
            p.predicates[sib].factors = factors;
            p.predicates[sib].siblings = siblings;
        }
        p.predicates[p.ndims - 1] = {siblings, factors, p.predicates[dim].restriction};
    }

    // step 2: mov (ndims-1) to (dim+1) position
    for (int d = p.ndims - 1; d > dim + 2; --d)
        prb_node_swap(p, d, d - 1);
}

void prb_node_swap(prb_t &p, int d0, int d1) {
    assert(d0 < p.ndims);
    assert(d1 < p.ndims);
    assert(p.ndims < max_ndims);

    if (d0 == d1) return;

    nstl::swap(p.nodes[d0], p.nodes[d1]);

    if (p.predicates.empty()) return;

    const auto &pd0 = p.predicates.find(d0);
    const auto &pd1 = p.predicates.find(d1);
    assert(pd0 != p.predicates.end());
    assert(pd1 != p.predicates.end());
    std::swap(pd0->second, pd1->second);

    for (auto &pr : p.predicates) {
        for (auto &sib: pr.second.siblings)
            if (sib == d0 || sib == d1)
                sib = d0 + d1 - sib;
    }
}

void prb_node_move(prb_t &p, int d0, int d1) {
    assert(d0 < p.ndims);
    assert(d1 < p.ndims);
    assert(p.ndims < max_ndims);

    if (d0 == d1) return;

    // TODO: enable faster version
#if 0
    node_t node = p.nodes[d0];

    if (d0 < d1)
        for (int d = d0; d < d1; ++d)
            p.nodes[d] = p.nodes[d + 1];
    else
        for (int d = d0; d > d1; --d)
            p.nodes[d] = p.nodes[d - 1];

    p.nodes[d1] = node;
#else
    int step = d0 < d1 ? +1 : -1;
    for (int d = d0; d != d1; d += step)
        prb_node_swap(p, d, d + step);
#endif
}

void prb_dump(const prb_t &p) {
    printf("@@@ type:%s:%s ndims:%d ", dnnl_dt2str(p.itype),
            dnnl_dt2str(p.otype), p.ndims);
    for (int d = 0; d < p.ndims; ++d)
        printf("[%zu:%td:%td:%td]", p.nodes[d].n, p.nodes[d].is, p.nodes[d].os,
                p.nodes[d].ss);
    printf(" off:%zu:%zu\n", p.ioff, p.ooff);

    for (auto &pr : p.predicates) {
        printf("[%d] : %zu ", pr.first, pr.second.restriction);
        for (size_t i = 0; i < pr.second.siblings.size(); ++i)
            printf("{sib:%d fac:%d} ", pr.second.siblings[i], pr.second.factors[i]);
        printf("\n");
    }
}

} // namespace tr

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
