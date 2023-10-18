/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "common/type_helpers.hpp"

#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

void fp8_emulation_e5m2_t::prepare_table() {
    host_->align(64);
    host_->L(label_table_to_f8_);
    // 0: mask for 2nd msb of f16 mantissa, used in rounding
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x0100);
    }
    // 64: mask for 1st msb of f16 mantissa, used to quiet nans
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x0200);
    }
    // 128: rounding bits of f16 mantissa
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x007f);
    }
    // 192: indices to pack high bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = u8 * 2 + 1; // 1, 3, 5, ...
        host_->db(idx);
    }
}

void fp8_emulation_e4m3_t::prepare_table() {
    host_->align(64);
    host_->L(label_table_from_f8_);
    // 0: map from f8_e4m3 byte to high byte of f16 (ignoring sign)
    for (uint8_t u8 = 0; u8 < 128; ++u8) {
        const float8_e4m3_t x8(u8, /* bit_cast = */ true);
        const float16_t x16 = static_cast<float16_t>(static_cast<float>(x8));
        const uint16_t u16 = x16.raw >> 8;
        host_->db(u16);
    }
    // 128: map from f8_e4m3 byte to low byte of f16 (ignoring sign)
    for (uint8_t u8 = 0; u8 < 128; ++u8) {
        const float8_e4m3_t x8(u8, /* bit_cast = */ true);
        const float16_t x16 = static_cast<float16_t>(static_cast<float>(x8));
        const uint16_t u16 = (x16.raw & 0xff);
        host_->db(u16);
    }
    // 256: indices to interleave high and low bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = (u8 / 2) + 64 * (u8 % 2); // 0, 64, 1, 65, ...
        host_->db(idx);
    }
    // 320: sign mask for fp8 values
    host_->dd(0x80808080);
    host_->align(64);
    host_->L(label_table_to_f8_);
    // 0: map from f16 sign+exponent word to f8_e4m3 sign+exponent byte
    static const uint16_t sign_exponent_table[64] = {// positive sign
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0008, 0x0010, 0x0018, 0x0020, 0x0028, 0x0030,
            0x0038, 0x0040, 0x0048, 0x0050, 0x0058, 0x0060, 0x0068, 0x0070,
            0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070,
            // negative sign
            0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,
            0x0080, 0x0080, 0x0088, 0x0090, 0x0098, 0x00a0, 0x00a8, 0x00b0,
            0x00b8, 0x00c0, 0x00c8, 0x00d0, 0x00d8, 0x00e0, 0x00e8, 0x00f0,
            0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0};
    for (int i = 0; i < 64; i++) {
        host_->dw(sign_exponent_table[i]);
    }
    // 128: map from f16 sign+exponent to f16 rounding shifter
    static const uint16_t rounding_shifter_table[32] = {0x4000, 0x4000, 0x4000,
            0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4400,
            0x4800, 0x4c00, 0x5000, 0x5400, 0x5800, 0x5c00, 0x6000, 0x6400,
            0x6800, 0x6c00, 0x7000, 0x7400, 0x7800, 0x7800, 0x7800, 0x7800,
            0x7800, 0x7800, 0x7800, 0x7800, 0x7800};
    for (int i = 0; i < 32; i++) {
        host_->dw(rounding_shifter_table[i]);
    }
    // 192: max f16 value that needs to be considered ie 480
    //      (anything greater will convert to same qnan in f8_e3m4)
    for (uint8_t u8 = 0; u8 < 16; ++u8) {
        const uint32_t f8_e4m3_max_threshold = 0x5f805f80;
        host_->dd(f8_e4m3_max_threshold);
    }
    // 256: indices to pack low bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = u8 * 2; // 0, 2, 4, ...
        host_->db(idx);
    }
    // 320: absolute value mask for f16 values
    host_->dd(0x7fff7fff);
}

void fp8_emulation_e5m2_t::vcvt_f8_to_f16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f8_e5m2
    host_->vpmovzxbw(xmm_out, op_in);
    host_->vpsllw(xmm_out, xmm_out, 8);
    // Floating point conversions typically set quiet bit for NaN inputs.
    // Here we add extra conversions to and from f32 to achieve this, but
    // in practice it should be okay to skip this step.
    // f32 <- f16
    host_->vcvtph2psx(xmm_out, xmm_out);
    // f16 <- f32
    host_->vcvtps2phx(xmm_out, xmm_out);
}

void fp8_emulation_e5m2_t::vcvt_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f8_e5m2
    // Floating point conversions typically set quiet bit for NaN inputs.
    // Here we skip this step as it will be handled during f32<-f16 stage.
    host_->vpmovzxbw(xmm_out, op_in);
    host_->vpsllw(xmm_out, xmm_out, 8);
    // f32 <- f16
    host_->vcvtph2psx(xmm_out, xmm_out);
}

void fp8_emulation_e4m3_t::vcvt_f8_to_f16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    host_->lea(reg64_aux_,
            host_->ptr[host_->rip + label_table_from_f8_]); // base of table
    // must use full Zmm to properly load all table values
    const Xbyak::Zmm zmm_in(
            op_in.isMEM() ? xmm_aux3_.getIdx() : op_in.getIdx());
    const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_aux2(xmm_aux2_.getIdx());
    const Xbyak::Zmm zmm_tmp(xmm_aux3_.getIdx());
    if (op_in.isMEM()) host_->vmovdqu8(zmm_in, op_in);
    // f16 <- f8_e4m3
    tabulate(data_type::f8_e4m3, zmm_aux1, zmm_in,
            host_->zword[reg64_aux_]); // high byte
    tabulate(data_type::f8_e4m3, zmm_aux2, zmm_in,
            host_->zword[reg64_aux_ + 128]); // low byte
    // sign correction
    // 0xf8 means A = A | (B & C)
    // 320 is offset of sign mask in table
    host_->vpternlogq(zmm_aux1, zmm_in, host_->ptr_b[reg64_aux_ + 320], 0xf8);
    // merge high and low
    host_->vmovdqu64(zmm_tmp, host_->zword[reg64_aux_ + 256]);
    host_->vpermt2b(zmm_aux2, zmm_tmp, zmm_aux1);

    host_->vmovdqu16(xmm_out, zmm_aux2);
}

void fp8_emulation_e4m3_t::vcvt_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f8_e4m3
    vcvt_f8_to_f16(xmm_out, op_in);

    // f32 <- f16
    host_->vcvtph2psx(xmm_out, xmm_out);
}

void fp8_emulation_e5m2_t::vcvt_f32_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f32
    host_->vcvtps2phx(xmm_out, op_in);
    // f8_e5m2 <- f16 (RNE)
    vcvt_f16_to_f8(xmm_out, xmm_out);
}

void fp8_emulation_e5m2_t::vcvt_f16_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // load base of table
    host_->lea(reg64_aux_, host_->ptr[host_->rip + label_table_to_f8_]);

    const Xbyak::Xmm xmm_in(
            op_in.isMEM() ? xmm_aux2_.getIdx() : op_in.getIdx());
    if (op_in.isMEM()) host_->vmovdqu16(xmm_in, op_in);

    // create mask from nan values
    host_->vfpclassph(kmask_aux_, xmm_in, 0x81);

    // get 2nd msb of mantissa and move to lsb for RNE tiebreaking
    host_->vpandd(xmm_aux1_, xmm_in, host_->ptr_b[reg64_aux_ + 0]);
    host_->vpsrlw(xmm_aux1_, xmm_aux1_, 8);

    // blend masks for nan handling and rounding
    host_->vmovdqu16(xmm_aux1_ | kmask_aux_, host_->ptr[reg64_aux_ + 64]);
    // set 1st msb of mantissa for nans (snan->qnan)
    // set 2nd msb of mantissa for RNE tiebreaking
    host_->vporq(xmm_out, xmm_aux1_, xmm_in);

    // perform rounding by adding rounding bits
    host_->vpaddw(xmm_aux1_, xmm_out, host_->ptr[reg64_aux_ + 128]);
    // reset nans
    host_->vmovdqu16(xmm_aux1_ | kmask_aux_, xmm_out);

    // extract the result (high byte of each word)
    // load odd indices from table
    host_->vmovdqu64(xmm_aux2_, host_->ptr[reg64_aux_ + 192]);
    // pack odd bytes to lower half of register
    // NOTE: there will be garbage in upper half of register
    host_->vpermb(xmm_out, xmm_aux2_, xmm_aux1_);
}

void fp8_emulation_e4m3_t::vcvt_f32_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f32
    host_->vcvtps2phx(xmm_out, op_in);
    // f8_e4m3 <- f16 (RNE)
    vcvt_f16_to_f8(xmm_out, xmm_out);
}

void fp8_emulation_e4m3_t::vcvt_f16_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    host_->lea(reg64_aux_,
            host_->ptr[host_->rip + label_table_to_f8_]); // load base of table

    const Xbyak::Xmm xmm_in(
            op_in.isMEM() ? xmm_aux3_.getIdx() : op_in.getIdx());
    if (op_in.isMEM()) host_->vmovdqu16(xmm_in, op_in);

    // get absolute value of f16 values
    host_->vpandd(xmm_aux1_, xmm_in, host_->ptr_b[reg64_aux_ + 320]);

    // isolate sign + exponent bits
    host_->vpsrlw(xmm_out, xmm_in, 10);

    const Xbyak::Zmm zmm_in(xmm_in.getIdx());
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());
    const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_aux2(xmm_aux2_.getIdx());
    // map from f16 sign + exponent to f8_e4m3 sign + exponent?
    tabulate(data_type::f16, zmm_aux2, zmm_out, host_->zword[reg64_aux_]);

    // map from f16 sign + exponent to rounding shifter
    // NOTE: must use full Zmm to access all table values
    host_->vpermw(zmm_out, zmm_out, host_->zword[reg64_aux_ + 128]);

    // all f16 values greater than some threshold will become nan,
    // so we take min w.r.t. this threshold
    host_->vmovdqu64(xmm_aux3_, host_->ptr[reg64_aux_ + 192]);
    host_->vpminuw(xmm_aux1_, xmm_aux1_, xmm_aux3_);

    // these additions will not overflow due to taking min above
    // NOTE: must use full Zmm to enable embedded rounding
    host_->vaddph(zmm_out | host_->T_rn_sae, zmm_aux1, zmm_out);
    host_->vpaddw(xmm_out, xmm_aux2_, xmm_out);

    // extract the result (low byte of each word)
    // load even indices from table
    host_->vmovdqu64(xmm_aux3_, host_->ptr[reg64_aux_ + 256]);
    // pack even bytes to lower half of register
    // NOTE: there will be garbage in upper half of register
    host_->vpermb(xmm_out, xmm_aux3_, xmm_out);
}

void fp8_emulation_e4m3_t::tabulate(const data_type_t dt,
        const Xbyak::Zmm &zmm_out, const Xbyak::Zmm &zmm_in,
        const Xbyak::Address &addr) {
    host_->vmovdqu64(zmm_out, addr);
    switch (dt) {
        case data_type::f8_e4m3:
            host_->vpermt2b(
                    zmm_out, zmm_in, host_->zword[addr.getRegExp() + 64]);
            break;
        case data_type::f16:
            host_->vpermt2w(
                    zmm_out, zmm_in, host_->zword[addr.getRegExp() + 64]);
            break;
        default: assert(!"Unsupported data type in helper routine");
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl