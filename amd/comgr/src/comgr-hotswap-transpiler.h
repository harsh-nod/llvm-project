// comgr-hotswap-transpiler.h — Cross-family ISA transpiler (GFX12 -> GFX9)
// Included inside the anonymous namespace of comgr-hotswap.cpp.
// Not a standalone compilation unit.

// ── NeedsTranspile ───────────────────────────────────────────────────────────

static bool NeedsTranspileImpl(const std::string &source_isa,
                               const std::string &target_isa) {
  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);
  if (src_cpu.empty() || tgt_cpu.empty()) return false;
  auto isGFX9 = [](const std::string &cpu) {
    return cpu.size() >= 4 && cpu[3] == '9';
  };
  auto isGFX12 = [](const std::string &cpu) {
    return cpu.size() >= 5 && cpu[3] == '1' && cpu[4] == '2';
  };
  return isGFX12(src_cpu) && isGFX9(tgt_cpu);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-Family ISA Transpiler (GFX12 → GFX9)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Mnemonic Translation Tables ──────────────────────────────────────────────

struct MnemonicMapping {
  const char* gfx12;
  const char* gfx9;
};

static const MnemonicMapping kGlobalMemMappings[] = {
    {"global_load_b32", "global_load_dword"},
    {"global_load_b64", "global_load_dwordx2"},
    {"global_load_b96", "global_load_dwordx3"},
    {"global_load_b128", "global_load_dwordx4"},
    {"global_load_u8", "global_load_ubyte"},
    {"global_load_i8", "global_load_sbyte"},
    {"global_load_u16", "global_load_ushort"},
    {"global_load_i16", "global_load_sshort"},
    {"global_load_d16_u8", "global_load_ubyte_d16"},
    {"global_load_d16_i8", "global_load_sbyte_d16"},
    {"global_load_d16_b16", "global_load_short_d16"},
    {"global_load_d16_hi_u8", "global_load_ubyte_d16_hi"},
    {"global_load_d16_hi_i8", "global_load_sbyte_d16_hi"},
    {"global_load_d16_hi_b16", "global_load_short_d16_hi"},
    {"global_store_b8", "global_store_byte"},
    {"global_store_b16", "global_store_short"},
    {"global_store_b32", "global_store_dword"},
    {"global_store_b64", "global_store_dwordx2"},
    {"global_store_b96", "global_store_dwordx3"},
    {"global_store_b128", "global_store_dwordx4"},
    {"global_load_addtid_b32", "global_load_dword_addtid"},
    {"global_store_addtid_b32", "global_store_dword_addtid"},
};

static const MnemonicMapping kFlatMemMappings[] = {
    {"flat_load_b32", "flat_load_dword"},
    {"flat_load_b64", "flat_load_dwordx2"},
    {"flat_load_b96", "flat_load_dwordx3"},
    {"flat_load_b128", "flat_load_dwordx4"},
    {"flat_load_u8", "flat_load_ubyte"},
    {"flat_load_i8", "flat_load_sbyte"},
    {"flat_load_u16", "flat_load_ushort"},
    {"flat_load_i16", "flat_load_sshort"},
    {"flat_load_d16_u8", "flat_load_ubyte_d16"},
    {"flat_load_d16_i8", "flat_load_sbyte_d16"},
    {"flat_load_d16_b16", "flat_load_short_d16"},
    {"flat_load_d16_hi_u8", "flat_load_ubyte_d16_hi"},
    {"flat_load_d16_hi_i8", "flat_load_sbyte_d16_hi"},
    {"flat_load_d16_hi_b16", "flat_load_short_d16_hi"},
    {"flat_store_b8", "flat_store_byte"},
    {"flat_store_b16", "flat_store_short"},
    {"flat_store_b32", "flat_store_dword"},
    {"flat_store_b64", "flat_store_dwordx2"},
    {"flat_store_b96", "flat_store_dwordx3"},
    {"flat_store_b128", "flat_store_dwordx4"},
};

static const MnemonicMapping kScratchMemMappings[] = {
    {"scratch_load_b32", "scratch_load_dword"},
    {"scratch_load_b64", "scratch_load_dwordx2"},
    {"scratch_load_b96", "scratch_load_dwordx3"},
    {"scratch_load_b128", "scratch_load_dwordx4"},
    {"scratch_load_u8", "scratch_load_ubyte"},
    {"scratch_load_i8", "scratch_load_sbyte"},
    {"scratch_load_u16", "scratch_load_ushort"},
    {"scratch_load_i16", "scratch_load_sshort"},
    {"scratch_load_d16_u8", "scratch_load_ubyte_d16"},
    {"scratch_load_d16_i8", "scratch_load_sbyte_d16"},
    {"scratch_load_d16_b16", "scratch_load_short_d16"},
    {"scratch_load_d16_hi_u8", "scratch_load_ubyte_d16_hi"},
    {"scratch_load_d16_hi_i8", "scratch_load_sbyte_d16_hi"},
    {"scratch_load_d16_hi_b16", "scratch_load_short_d16_hi"},
    {"scratch_store_b8", "scratch_store_byte"},
    {"scratch_store_b16", "scratch_store_short"},
    {"scratch_store_b32", "scratch_store_dword"},
    {"scratch_store_b64", "scratch_store_dwordx2"},
    {"scratch_store_b96", "scratch_store_dwordx3"},
    {"scratch_store_b128", "scratch_store_dwordx4"},
};

static const MnemonicMapping kBufferMemMappings[] = {
    {"buffer_load_b32", "buffer_load_dword"},
    {"buffer_load_b64", "buffer_load_dwordx2"},
    {"buffer_load_b96", "buffer_load_dwordx3"},
    {"buffer_load_b128", "buffer_load_dwordx4"},
    {"buffer_load_u8", "buffer_load_ubyte"},
    {"buffer_load_i8", "buffer_load_sbyte"},
    {"buffer_load_u16", "buffer_load_ushort"},
    {"buffer_load_i16", "buffer_load_sshort"},
    {"buffer_load_d16_u8", "buffer_load_ubyte_d16"},
    {"buffer_load_d16_i8", "buffer_load_sbyte_d16"},
    {"buffer_load_d16_b16", "buffer_load_short_d16"},
    {"buffer_load_d16_hi_u8", "buffer_load_ubyte_d16_hi"},
    {"buffer_load_d16_hi_i8", "buffer_load_sbyte_d16_hi"},
    {"buffer_load_d16_hi_b16", "buffer_load_short_d16_hi"},
    {"buffer_store_b8", "buffer_store_byte"},
    {"buffer_store_b16", "buffer_store_short"},
    {"buffer_store_b32", "buffer_store_dword"},
    {"buffer_store_b64", "buffer_store_dwordx2"},
    {"buffer_store_b96", "buffer_store_dwordx3"},
    {"buffer_store_b128", "buffer_store_dwordx4"},
};

static const MnemonicMapping kDSMappings[] = {
    {"ds_load_b32", "ds_read_b32"},
    {"ds_load_b64", "ds_read_b64"},
    {"ds_load_b96", "ds_read_b96"},
    {"ds_load_b128", "ds_read_b128"},
    {"ds_load_u8", "ds_read_u8"},
    {"ds_load_i8", "ds_read_i8"},
    {"ds_load_u16", "ds_read_u16"},
    {"ds_load_i16", "ds_read_i16"},
    {"ds_load_2addr_b32", "ds_read2_b32"},
    {"ds_load_2addr_stride64_b32", "ds_read2st64_b32"},
    {"ds_load_2addr_b64", "ds_read2_b64"},
    {"ds_load_2addr_stride64_b64", "ds_read2st64_b64"},
    {"ds_load_u8_d16", "ds_read_u8_d16"},
    {"ds_load_i8_d16", "ds_read_i8_d16"},
    {"ds_load_u16_d16", "ds_read_u16_d16"},
    {"ds_load_u8_d16_hi", "ds_read_u8_d16_hi"},
    {"ds_load_i8_d16_hi", "ds_read_i8_d16_hi"},
    {"ds_load_u16_d16_hi", "ds_read_u16_d16_hi"},
    {"ds_load_addtid_b32", "ds_read_addtid_b32"},
    {"ds_store_b32", "ds_write_b32"},
    {"ds_store_b64", "ds_write_b64"},
    {"ds_store_b128", "ds_write_b128"},
    {"ds_store_2addr_b32", "ds_write2_b32"},
    {"ds_store_2addr_stride64_b32", "ds_write2st64_b32"},
    {"ds_store_b8", "ds_write_b8"},
    {"ds_store_b16", "ds_write_b16"},
};

static const MnemonicMapping kSMEMMappings[] = {
    {"s_load_b32", "s_load_dword"},
    {"s_load_b64", "s_load_dwordx2"},
    {"s_load_b96", "s_load_dwordx3"},
    {"s_load_b128", "s_load_dwordx4"},
    {"s_load_b256", "s_load_dwordx8"},
    {"s_load_b512", "s_load_dwordx16"},
    {"s_store_b32", "s_store_dword"},
    {"s_store_b64", "s_store_dwordx2"},
    {"s_store_b128", "s_store_dwordx4"},
    {"s_buffer_load_b32", "s_buffer_load_dword"},
    {"s_buffer_load_b64", "s_buffer_load_dwordx2"},
    {"s_buffer_load_b128", "s_buffer_load_dwordx4"},
    {"s_buffer_load_b256", "s_buffer_load_dwordx8"},
    {"s_buffer_load_b512", "s_buffer_load_dwordx16"},
};

static const MnemonicMapping kScalarALURenames[] = {
    {"s_add_co_u32", "s_add_u32"},
    {"s_sub_co_u32", "s_sub_u32"},
    {"s_add_co_ci_u32", "s_addc_u32"},
    {"s_sub_co_ci_u32", "s_subb_u32"},
    {"s_add_co_i32", "s_add_i32"},
    {"s_sub_co_i32", "s_sub_i32"},
    {"s_and_not1_b32", "s_andn2_b32"},
    {"s_and_not1_b64", "s_andn2_b64"},
    {"s_or_not1_b32", "s_orn2_b32"},
    {"s_or_not1_b64", "s_orn2_b64"},
};

static const MnemonicMapping kVALURenames[] = {
    {"v_max_num_f32", "v_max_f32"},
    {"v_min_num_f32", "v_min_f32"},
    {"v_max_num_f16", "v_max_f16"},
    {"v_min_num_f16", "v_min_f16"},
    {"v_max_num_f64", "v_max_f64"},
    {"v_min_num_f64", "v_min_f64"},
    {"v_maxmin_num_f32", "v_maxmin_f32"},
    {"v_minmax_num_f32", "v_minmax_f32"},
    {"v_maxmin_num_f16", "v_maxmin_f16"},
    {"v_minmax_num_f16", "v_minmax_f16"},
    {"v_add_nc_u32", "v_add_u32"},
    {"v_sub_nc_u32", "v_sub_u32"},
    {"v_add_nc_i32", "v_add_i32"},
    {"v_sub_nc_i32", "v_sub_i32"},
};

static const MnemonicMapping kGlobalAtomicRenames[] = {
    {"global_atomic_add_u32", "global_atomic_add"},
    {"global_atomic_sub_u32", "global_atomic_sub"},
    {"global_atomic_and_b32", "global_atomic_and"},
    {"global_atomic_or_b32", "global_atomic_or"},
    {"global_atomic_xor_b32", "global_atomic_xor"},
    {"global_atomic_min_i32", "global_atomic_smin"},
    {"global_atomic_max_i32", "global_atomic_smax"},
    {"global_atomic_min_u32", "global_atomic_umin"},
    {"global_atomic_max_u32", "global_atomic_umax"},
    {"global_atomic_swap_b32", "global_atomic_swap"},
    {"global_atomic_cmpswap_b32", "global_atomic_cmpswap"},
    {"global_atomic_add_u64", "global_atomic_add_x2"},
    {"global_atomic_sub_u64", "global_atomic_sub_x2"},
    {"global_atomic_and_b64", "global_atomic_and_x2"},
    {"global_atomic_or_b64", "global_atomic_or_x2"},
    {"global_atomic_xor_b64", "global_atomic_xor_x2"},
    {"global_atomic_swap_b64", "global_atomic_swap_x2"},
    {"global_atomic_cmpswap_b64", "global_atomic_cmpswap_x2"},
    {"global_atomic_add_f32", "global_atomic_add_f32"},
    {"global_atomic_pk_add_f16", "global_atomic_pk_add_f16"},
};

static const MnemonicMapping kFlatAtomicRenames[] = {
    {"flat_atomic_add_u32", "flat_atomic_add"},
    {"flat_atomic_sub_u32", "flat_atomic_sub"},
    {"flat_atomic_and_b32", "flat_atomic_and"},
    {"flat_atomic_or_b32", "flat_atomic_or"},
    {"flat_atomic_xor_b32", "flat_atomic_xor"},
    {"flat_atomic_min_i32", "flat_atomic_smin"},
    {"flat_atomic_max_i32", "flat_atomic_smax"},
    {"flat_atomic_min_u32", "flat_atomic_umin"},
    {"flat_atomic_max_u32", "flat_atomic_umax"},
    {"flat_atomic_swap_b32", "flat_atomic_swap"},
    {"flat_atomic_cmpswap_b32", "flat_atomic_cmpswap"},
    {"flat_atomic_add_u64", "flat_atomic_add_x2"},
    {"flat_atomic_sub_u64", "flat_atomic_sub_x2"},
    {"flat_atomic_swap_b64", "flat_atomic_swap_x2"},
    {"flat_atomic_cmpswap_b64", "flat_atomic_cmpswap_x2"},
};

static const MnemonicMapping kDSAtomicRenames[] = {
    {"ds_add_u32", "ds_add_u32"},
    {"ds_add_rtn_u32", "ds_add_rtn_u32"},
    {"ds_cmpstore_b32", "ds_cmpst_b32"},
    {"ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32"},
    {"ds_cmpstore_b64", "ds_cmpst_b64"},
    {"ds_cmpstore_rtn_b64", "ds_cmpst_rtn_b64"},
};

static std::unordered_map<std::string, std::string> BuildMnemonicMap() {
  std::unordered_map<std::string, std::string> map;
  auto addMappings = [&](const MnemonicMapping* mappings, size_t count) {
    for (size_t i = 0; i < count; ++i)
      map[mappings[i].gfx12] = mappings[i].gfx9;
  };
  addMappings(kGlobalMemMappings, sizeof(kGlobalMemMappings) / sizeof(kGlobalMemMappings[0]));
  addMappings(kFlatMemMappings, sizeof(kFlatMemMappings) / sizeof(kFlatMemMappings[0]));
  addMappings(kScratchMemMappings, sizeof(kScratchMemMappings) / sizeof(kScratchMemMappings[0]));
  addMappings(kBufferMemMappings, sizeof(kBufferMemMappings) / sizeof(kBufferMemMappings[0]));
  addMappings(kDSMappings, sizeof(kDSMappings) / sizeof(kDSMappings[0]));
  addMappings(kSMEMMappings, sizeof(kSMEMMappings) / sizeof(kSMEMMappings[0]));
  addMappings(kScalarALURenames, sizeof(kScalarALURenames) / sizeof(kScalarALURenames[0]));
  addMappings(kVALURenames, sizeof(kVALURenames) / sizeof(kVALURenames[0]));
  addMappings(kGlobalAtomicRenames, sizeof(kGlobalAtomicRenames) / sizeof(kGlobalAtomicRenames[0]));
  addMappings(kFlatAtomicRenames, sizeof(kFlatAtomicRenames) / sizeof(kFlatAtomicRenames[0]));
  addMappings(kDSAtomicRenames, sizeof(kDSAtomicRenames) / sizeof(kDSAtomicRenames[0]));
  return map;
}

static const std::unordered_map<std::string, std::string>& GetMnemonicMap() {
  static auto map = BuildMnemonicMap();
  return map;
}

// ── Wave32→Wave64 EXEC Patterns ─────────────────────────────────────────────

static bool WritesExecLo(const std::string& line) {
  size_t mnem_end = line.find_first_of(" \t");
  if (mnem_end == std::string::npos) return false;
  size_t op_start = line.find_first_not_of(" \t,", mnem_end);
  if (op_start == std::string::npos) return false;
  if (line.compare(op_start, 7, "exec_lo") == 0) return true;
  std::string mnemonic = line.substr(0, mnem_end);
  if (mnemonic.find("saveexec_b32") != std::string::npos) return true;
  return false;
}

// ── Wait Counter Translation ─────────────────────────────────────────────────

static bool IsWaitInstruction(const std::string& mnemonic) {
  return mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_storecnt" ||
         mnemonic == "s_wait_samplecnt" || mnemonic == "s_wait_bvhcnt" ||
         mnemonic == "s_wait_expcnt" || mnemonic == "s_wait_dscnt" ||
         mnemonic == "s_wait_kmcnt" || mnemonic == "s_wait_loadcnt_dscnt" ||
         mnemonic == "s_wait_storecnt_dscnt" || mnemonic == "s_wait_xcnt" ||
         mnemonic == "s_wait_asynccnt" || mnemonic == "s_wait_tensorcnt";
}

static std::string TranslateWaitInstruction(const std::string& line) {
  std::string mnemonic;
  int count = 0;
  std::istringstream iss(line);
  iss >> mnemonic >> count;
  if (iss.fail()) count = 0;

  if (mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_samplecnt" ||
      mnemonic == "s_wait_bvhcnt" || mnemonic == "s_wait_storecnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_dscnt" || mnemonic == "s_wait_kmcnt")
    return "s_waitcnt lgkmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_expcnt")
    return "s_waitcnt expcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_loadcnt_dscnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  if (mnemonic == "s_wait_storecnt_dscnt")
    return "s_waitcnt vmcnt(" + std::to_string(count) +
           ") lgkmcnt(" + std::to_string(count) + ")";
  return "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)";
}

// ── Unsupported Instruction Detection ────────────────────────────────────────

static bool IsUnsupportedOnGFX9(const std::string& mnemonic) {
  if (mnemonic.find("tensor_") == 0) return true;
  if (mnemonic.find("cluster_") == 0) return true;
  if (mnemonic.find("_prefetch_") != std::string::npos) return true;
  if (mnemonic.find("v_permlane16") == 0) return true;
  if (mnemonic.find("v_permlanex16") == 0) return true;
  if (mnemonic == "s_wait_alu") return true;
  if (mnemonic == "s_delay_alu") return true;
  return false;
}

// ── VCC Register Width Translation ───────────────────────────────────────────

static std::string WidenVccReferences(const std::string& line) {
  std::string result = line;
  size_t mnem_end = result.find_first_of(" \t");
  if (mnem_end == std::string::npos) return result;
  std::string operands = result.substr(mnem_end);
  size_t pos = 0;
  while ((pos = operands.find("vcc_lo", pos)) != std::string::npos) {
    size_t end = pos + 6;
    if (end < operands.size() && (std::isalnum(operands[end]) || operands[end] == '_')) {
      pos = end;
      continue;
    }
    operands.replace(pos, 6, "vcc");
    pos += 3;
  }
  return result.substr(0, mnem_end) + operands;
}

// ── EXEC Width Widening ──────────────────────────────────────────────────────

static std::vector<std::string> WidenExecOperation(const std::string& line, bool compact_mode = false) {
  std::vector<std::string> result;
  std::string mnemonic = line.substr(0, line.find_first_of(" \t"));

  if (mnemonic.find("saveexec_b32") != std::string::npos) {
    std::string b64_mnem = mnemonic;
    size_t b32_pos = b64_mnem.find("_b32");
    b64_mnem.replace(b32_pos, 4, "_b64");
    size_t not1_pos = b64_mnem.find("_not1_");
    if (not1_pos != std::string::npos)
      b64_mnem.replace(not1_pos, 6, "n2_");

    std::string ops_part = line.substr(line.find_first_of(" \t"));
    size_t op_start = ops_part.find_first_not_of(" \t");
    if (op_start != std::string::npos) {
      std::string ops = ops_part.substr(op_start);
      size_t comma = ops.find(',');
      if (comma != std::string::npos) {
        std::string dst = ops.substr(0, comma);
        size_t ds = dst.find_first_not_of(" \t");
        size_t de = dst.find_last_not_of(" \t");
        dst = dst.substr(ds, de - ds + 1);
        std::string src = ops.substr(comma + 1);
        size_t ss = src.find_first_not_of(" \t");
        src = src.substr(ss);

        size_t vcc_pos = src.find("vcc_lo");
        if (vcc_pos != std::string::npos)
          src.replace(vcc_pos, 6, "vcc");

        if (dst[0] == 's' && dst.size() > 1 && std::isdigit(dst[1])) {
          std::string src32 = src;
          if (src32 == "vcc") src32 = "vcc_lo";
          result.push_back("s_mov_b32 " + dst + ", exec_lo");
          bool is_or = (b64_mnem.find("s_or_saveexec") == 0);
          if (is_or)
            result.push_back("s_or_b32 exec_lo, exec_lo, " + src32);
          else if (b64_mnem.find("andn2") != std::string::npos)
            result.push_back("s_andn2_b32 exec_lo, exec_lo, " + src32);
          else
            result.push_back("s_and_b32 exec_lo, exec_lo, " + src32);
          if (is_or || !compact_mode)
            result.push_back("s_mov_b32 exec_hi, 0");
          return result;
        }
        result.push_back(b64_mnem + " " + dst + ", " + src);
        result.push_back("s_mov_b32 exec_hi, 0");
        return result;
      }
    }
    result.push_back(b64_mnem + ops_part);
    result.push_back("s_mov_b32 exec_hi, 0");
    return result;
  }

  result.push_back(line);
  if (WritesExecLo(line))
    result.push_back("s_mov_b32 exec_hi, 0");
  return result;
}

// ── Operand Syntax Translation ───────────────────────────────────────────────

static std::string TranslateOperandSyntax(const std::string& line,
                                           const std::string& mnemonic) {
  (void)mnemonic;
  std::string result = line;
  {
    size_t pos = result.find("scope:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      result.erase(pos, end - pos);
    }
  }
  {
    size_t pos = result.find("th:");
    if (pos != std::string::npos) {
      size_t end = result.find_first_of(" \t,", pos);
      if (end == std::string::npos) end = result.size();
      std::string th_value = result.substr(pos, end - pos);
      if (th_value.find("TH_ATOMIC_RETURN") != std::string::npos)
        result.replace(pos, end - pos, "sc0");
      else
        result.erase(pos, end - pos);
    }
  }
  {
    size_t pos = result.find(" nv");
    while (pos != std::string::npos) {
      size_t end = pos + 3;
      if (end >= result.size() || result[end] == ' ' || result[end] == '\t' ||
          result[end] == ',' || result[end] == '\0') {
        result.erase(pos, end - pos);
      } else {
        pos = result.find(" nv", pos + 1);
        continue;
      }
      pos = result.find(" nv", pos);
    }
  }
  {
    size_t pos = result.find("scale_offset");
    if (pos != std::string::npos) {
      size_t end = pos + 12;
      if (pos > 0 && (result[pos-1] == ' ' || result[pos-1] == ',')) --pos;
      result.erase(pos, end - pos);
    }
  }
  while (!result.empty() && (result.back() == ' ' || result.back() == '\t' ||
                              result.back() == ','))
    result.pop_back();
  return result;
}

// ── Extract/Replace Mnemonic ─────────────────────────────────────────────────

static std::string TranspileExtractMnemonic(const std::string& line) {
  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) return "";
  size_t end = line.find_first_of(" \t", start);
  if (end == std::string::npos) return line.substr(start);
  return line.substr(start, end - start);
}

static std::string TranspileReplaceMnemonic(const std::string& line,
                                             const std::string& old_mnemonic,
                                             const std::string& new_mnemonic) {
  size_t pos = line.find(old_mnemonic);
  if (pos == std::string::npos) return line;
  std::string result = line;
  result.replace(pos, old_mnemonic.size(), new_mnemonic);
  return result;
}

// ── TTMP Taint Analysis ──────────────────────────────────────────────────────

enum class RegKind { SGPR, VGPR, TTMP, SCC, VCC, EXEC, Other };

static RegKind ClassifyReg(unsigned reg, const llvm::MCRegisterInfo& MRI) {
  const char* name = MRI.getName(reg);
  if (!name) return RegKind::Other;
  if (strncmp(name, "TTMP", 4) == 0) return RegKind::TTMP;
  if (strncmp(name, "SGPR", 4) == 0) return RegKind::SGPR;
  if (strncmp(name, "VGPR", 4) == 0) return RegKind::VGPR;
  if (strcmp(name, "SCC") == 0) return RegKind::SCC;
  if (strncmp(name, "VCC", 3) == 0) return RegKind::VCC;
  if (strncmp(name, "EXEC", 4) == 0) return RegKind::EXEC;
  return RegKind::Other;
}

static bool IsRegTainted(unsigned reg, const std::set<unsigned>& tainted,
                         const llvm::MCRegisterInfo& MRI) {
  if (tainted.count(reg)) return true;
  for (auto sub : MRI.subregs(reg))
    if (tainted.count(sub)) return true;
  for (auto sup : MRI.superregs(reg))
    if (tainted.count(sup)) return true;
  return false;
}

static void TaintReg(unsigned reg, std::set<unsigned>& tainted,
                     const llvm::MCRegisterInfo& MRI) {
  tainted.insert(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.insert(sub);
}

static void UntaintReg(unsigned reg, std::set<unsigned>& tainted,
                       const llvm::MCRegisterInfo& MRI) {
  tainted.erase(reg);
  for (auto sub : MRI.subregs(reg))
    tainted.erase(sub);
  for (auto sup : MRI.superregs(reg))
    tainted.erase(sup);
}

enum class TaintAction { Keep, Skip, Replace };

struct TaintResult {
  TaintAction action;
  std::string replace_dst;
  std::string replace_src;
};

static void GetInstRegs(const llvm::MCInst& inst,
                        const llvm::MCInstrInfo& MCII,
                        const llvm::MCRegisterInfo& MRI,
                        std::vector<unsigned>& defs,
                        std::vector<unsigned>& uses) {
  (void)MRI;
  const llvm::MCInstrDesc& desc = MCII.get(inst.getOpcode());
  unsigned num_defs = desc.getNumDefs();
  for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
    const auto& op = inst.getOperand(i);
    if (!op.isReg() || op.getReg() == 0) continue;
    if (i < num_defs)
      defs.push_back(op.getReg());
    else
      uses.push_back(op.getReg());
  }
  for (auto imp : desc.implicit_defs())
    defs.push_back(imp);
  for (auto imp : desc.implicit_uses())
    uses.push_back(imp);
}

struct SourceInstrForTaint {
  std::string text;
  llvm::MCInst inst;
  bool valid_inst;
};

static std::vector<TaintResult> AnalyzeTTMPTaint(
    const std::vector<SourceInstrForTaint>& instrs,
    const llvm::MCInstrInfo& MCII,
    const llvm::MCRegisterInfo& MRI) {

  std::vector<TaintResult> results;
  results.reserve(instrs.size());
  std::set<unsigned> tainted;
  bool dump = std::getenv("HSA_HOTSWAP_DUMP") != nullptr;

  for (size_t i = 0; i < instrs.size(); ++i) {
    const auto& si = instrs[i];
    const auto& text = si.text;
    std::string mnemonic = TranspileExtractMnemonic(text);

    TaintResult tr;
    tr.action = TaintAction::Keep;

    if (!si.valid_inst) {
      results.push_back(tr);
      continue;
    }

    if (mnemonic.empty() || mnemonic[0] != 's' ||
        mnemonic.find("s_cbranch_") == 0 || mnemonic == "s_branch" ||
        mnemonic == "s_endpgm" || mnemonic == "s_barrier" ||
        mnemonic.find("s_barrier_") == 0 || mnemonic == "s_nop" ||
        mnemonic == "s_waitcnt" || mnemonic.find("s_wait_") == 0 ||
        mnemonic == "s_clause" || mnemonic == "s_delay_alu" ||
        mnemonic == "s_wait_alu" || mnemonic == "s_code_end" ||
        mnemonic == "s_set_inst_prefetch_distance") {
      results.push_back(tr);
      continue;
    }

    std::vector<unsigned> defs, uses;
    GetInstRegs(si.inst, MCII, MRI, defs, uses);

    bool uses_ttmp = false;
    for (auto r : uses)
      if (ClassifyReg(r, MRI) == RegKind::TTMP) { uses_ttmp = true; break; }
    bool defs_ttmp = false;
    for (auto r : defs)
      if (ClassifyReg(r, MRI) == RegKind::TTMP) { defs_ttmp = true; break; }

    if (mnemonic == "s_getreg_b32" &&
        text.find("HW_REG_IB_STS2") != std::string::npos) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) TaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_IB_STS2): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32") &&
        text.find("HW_REG_WAVE_MODE") != std::string::npos) {
      tr.action = TaintAction::Skip;
      if (dump) std::cerr << "hotswap: taint: SKIP (HW_REG_WAVE_MODE): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (uses_ttmp || defs_ttmp) {
      if (mnemonic == "s_cselect_b32") {
        tr.action = TaintAction::Replace;
        size_t op_start = text.find(mnemonic) + mnemonic.size();
        std::string ops = text.substr(op_start);
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos)
          tr.replace_dst = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        tr.replace_src = (text.find("ttmp9") != std::string::npos) ? "v5" : "v4";
        for (auto r : defs) UntaintReg(r, tainted, MRI);
        for (auto r : uses) {
          if (ClassifyReg(r, MRI) == RegKind::SCC)
            UntaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: REPLACE (s_cselect ttmp → "
                            << tr.replace_dst << " = " << tr.replace_src << "): " << text << "\n";
      } else {
        tr.action = TaintAction::Skip;
        for (auto r : defs) {
          RegKind kind = ClassifyReg(r, MRI);
          if (kind == RegKind::SGPR || kind == RegKind::SCC)
            TaintReg(r, tainted, MRI);
        }
        if (dump) std::cerr << "hotswap: taint: SKIP (direct TTMP): " << text << "\n";
      }
      results.push_back(tr);
      continue;
    }

    if (mnemonic.find("s_load_") == 0 || mnemonic.find("s_buffer_load_") == 0) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump && !tainted.empty())
        std::cerr << "hotswap: taint: KEEP (s_load clears taint on defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (mnemonic.find("s_cmp_") == 0) {
      bool any_tainted = false;
      for (auto r : uses)
        if (IsRegTainted(r, tainted, MRI)) { any_tainted = true; break; }
      if (any_tainted) {
        tr.action = TaintAction::Skip;
        for (auto r : defs) TaintReg(r, tainted, MRI);
        if (dump) std::cerr << "hotswap: taint: SKIP (s_cmp tainted): " << text << "\n";
        results.push_back(tr);
        continue;
      }
      results.push_back(tr);
      continue;
    }

    bool has_tainted_src = false;
    bool has_untainted_sgpr_src = false;
    for (auto r : uses) {
      RegKind kind = ClassifyReg(r, MRI);
      if (kind == RegKind::SGPR || kind == RegKind::SCC) {
        if (IsRegTainted(r, tainted, MRI))
          has_tainted_src = true;
        else
          has_untainted_sgpr_src = true;
      }
    }

    if (has_tainted_src && !has_untainted_sgpr_src) {
      tr.action = TaintAction::Skip;
      for (auto r : defs) {
        RegKind kind = ClassifyReg(r, MRI);
        if (kind == RegKind::SGPR || kind == RegKind::SCC)
          TaintReg(r, tainted, MRI);
      }
      if (dump) std::cerr << "hotswap: taint: SKIP (all srcs tainted): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (has_tainted_src && has_untainted_sgpr_src) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
      if (dump) std::cerr << "hotswap: taint: KEEP (mixed taint, clear defs): " << text << "\n";
      results.push_back(tr);
      continue;
    }

    if (!tainted.empty()) {
      for (auto r : defs) UntaintReg(r, tainted, MRI);
    }
    results.push_back(tr);
  }

  return results;
}

// ── Kernel Descriptor Patching for Wave64 ────────────────────────────────────

static void PatchKernelDescriptorsForWave64(uint8_t* elf, size_t elf_size,
                                             const ElfInfo& info) {
  if (info.text_idx < 0) return;
  const uint8_t* text = elf + info.text_offset;

  for (uint64_t offset = 0; offset + 64 <= info.text_size; offset += 256) {
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + offset + 16, 8);
    if (entry_offset != 256) continue;

    uint32_t rsrc1;
    std::memcpy(&rsrc1, text + offset + 48, 4);
    rsrc1 &= 0x00FFFFFFu;
    rsrc1 |= (1u << 21) | (1u << 23);

    uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
    uint32_t sgpr_field12_kd = (rsrc1 >> 6) & 0x3Fu;
    uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
    if (num_vgprs < 8u) num_vgprs = 8u;
    num_vgprs += 4u;
    uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;
    if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
    uint32_t rsrc1_vgpr_field = gfx9_vgpr + 1u;
    if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
    rsrc1 &= ~0xFFFu;
    rsrc1 |= (rsrc1_vgpr_field << 6u);
    {
      uint32_t orig_sgpr_field = sgpr_field12_kd;
      uint32_t num_sgprs = (orig_sgpr_field + 1u) * 16u + 8u;
      const char* sgpr_key = ".sgpr_count";
      for (size_t i = 0; i + 12 < elf_size; i++) {
        if (std::memcmp(elf + i, sgpr_key, 11) == 0) {
          uint8_t val = elf[i + 11];
          uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? elf[i+12] : 0);
          if (sc + 8 > num_sgprs) num_sgprs = sc + 8;
          break;
        }
      }
      uint32_t gfx9_sgpr = (num_sgprs / 8u) - 1u;
      if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
      rsrc1 |= gfx9_sgpr;
    }
    std::memcpy(elf + info.text_offset + offset + 48, &rsrc1, 4);

    uint32_t rsrc2;
    std::memcpy(&rsrc2, text + offset + 52, 4);
    rsrc2 |= (1u << 7);
    rsrc2 |= (1u << 8);
    rsrc2 |= (1u << 9);
    std::memcpy(elf + info.text_offset + offset + 52, &rsrc2, 4);

    uint16_t props;
    std::memcpy(&props, text + offset + 56, 2);
    props = static_cast<uint16_t>((static_cast<uint32_t>(props) & ~(1u << 10)) &
                                  0xFFFFu);
    std::memcpy(elf + info.text_offset + offset + 56, &props, 2);

    uint32_t rsrc3 = gfx9_vgpr;
    std::memcpy(elf + info.text_offset + offset + 44, &rsrc3, 4);
  }
}

// ── ELF Metadata Patching ────────────────────────────────────────────────────

static void PatchElfMetadata(uint8_t* elf, size_t elf_size,
                              const std::string& target_cpu) {
  uint32_t e_flags;
  std::memcpy(&e_flags, elf + 48, 4);
  uint8_t target_mach = 0;
  if (target_cpu == "gfx950") target_mach = 0x4f;
  else if (target_cpu == "gfx942") target_mach = 0x4c;
  else if (target_cpu == "gfx90a") target_mach = 0x42;
  if (target_mach != 0) {
    e_flags = (e_flags & ~0xFFu) | target_mach;
    std::memcpy(elf + 48, &e_flags, 4);
  }

  std::string old_isa_full = "amdgcn-amd-amdhsa--gfx1250";
  std::string new_isa_full = "amdgcn-amd-amdhsa--" + target_cpu;
  for (size_t i = 0; i + old_isa_full.size() <= elf_size; ++i) {
    if (std::memcmp(elf + i, old_isa_full.data(), old_isa_full.size()) == 0) {
      if (new_isa_full.size() <= old_isa_full.size()) {
        std::memcpy(elf + i, new_isa_full.data(), new_isa_full.size());
        for (size_t j = new_isa_full.size(); j < old_isa_full.size(); ++j)
          elf[i + j] = ' ';
      }
    }
  }

  for (size_t i = 0; i + 7 <= elf_size; ++i) {
    if (std::memcmp(elf + i, "gfx1250", 7) == 0) {
      if (target_cpu.size() <= 7) {
        std::memcpy(elf + i, target_cpu.c_str(), target_cpu.size());
        for (size_t j = target_cpu.size(); j < 7; ++j)
          elf[i + j] = '0';
      }
    }
  }

  {
    const char* wf_key = ".wavefront_size";
    size_t wf_key_len = 15;
    for (size_t i = 0; i + wf_key_len + 1 <= elf_size; ++i) {
      if (std::memcmp(elf + i, wf_key, wf_key_len) == 0) {
        uint8_t val = elf[i + wf_key_len];
        if (val == 0x20) {
          elf[i + wf_key_len] = 0x40;
          std::cerr << "hotswap: transpile: patched wavefront_size 32 → 64\n";
        }
      }
    }
  }
}


// Forward declaration for recursive dispatch from HandleVOPDInstruction
static std::vector<std::string> TranslateInstruction(const std::string& asm_line,
                                               const std::string& source_cpu,
                                               const std::string& target_cpu,
                                               int scale_temp_vgpr,
                                               int cmpx_temp_sgpr,
                                               bool compact_mode);

// ── Operand parsing helper ───────────────────────────────────────────────────

static std::vector<std::string> ParseOperandList(const std::string& line,
                                                  const std::string& mnemonic) {
  std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
  size_t op_start = ops.find_first_not_of(" \t");
  if (op_start != std::string::npos) ops = ops.substr(op_start);
  std::vector<std::string> operands;
  std::istringstream oss(ops);
  std::string tok;
  while (std::getline(oss, tok, ',')) {
    size_t s = tok.find_first_not_of(" \t");
    size_t e = tok.find_last_not_of(" \t");
    if (s != std::string::npos)
      operands.push_back(tok.substr(s, e - s + 1));
  }
  return operands;
}

// ── Translation result type ──────────────────────────────────────────────────
// nullopt = not handled (try next handler); has_value() = handled (may be empty for skip)

using TranslationResult = std::optional<std::vector<std::string>>;

// ── Handler: Wait Instructions ───────────────────────────────────────────────

static TranslationResult HandleWaitInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  if (IsWaitInstruction(mnemonic))
    return std::vector<std::string>{TranslateWaitInstruction(line)};
  if (mnemonic == "s_wait_alu" || mnemonic == "s_delay_alu" ||
      mnemonic == "s_clause" || mnemonic == "s_set_inst_prefetch_distance")
    return std::vector<std::string>{};
  return std::nullopt;
}

// ── Handler: Unsupported / Skip Instructions ─────────────────────────────────

static TranslationResult HandleUnsupportedInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  if ((mnemonic == "s_setreg_imm32_b32" || mnemonic == "s_setreg_b32" ||
       mnemonic == "s_getreg_b32") &&
      (line.find("HW_REG_WAVE_MODE") != std::string::npos ||
       line.find("HW_REG_IB_STS2") != std::string::npos))
    return std::vector<std::string>{};
  if (line.find("ttmp6") != std::string::npos ||
      line.find("ttmp7") != std::string::npos ||
      line.find("ttmp9") != std::string::npos)
    return std::vector<std::string>{};
  if (mnemonic == "s_code_end")
    return std::vector<std::string>{};
  if (mnemonic == "s_endpgm")
    return std::vector<std::string>{".L_exit:", "s_endpgm"};
  if (mnemonic == "s_barrier_signal")
    return std::vector<std::string>{"s_barrier"};
  if (mnemonic == "s_barrier_wait")
    return std::vector<std::string>{"s_nop 0"};
  if (IsUnsupportedOnGFX9(mnemonic))
    return std::vector<std::string>{"s_nop 0 ; UNSUPPORTED: " + mnemonic};
  return std::nullopt;
}

// ── Handler: SMEM Instructions ───────────────────────────────────────────────

static TranslationResult HandleSMEMInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  if (mnemonic != "s_load_b96") return std::nullopt;
  std::vector<std::string> result;
  std::regex reg_range(R"(s\[(\d+):(\d+)\])");
  std::smatch m;
  std::string ops_part = line.substr(line.find(mnemonic) + mnemonic.size());
  if (std::regex_search(ops_part, m, reg_range)) {
    int lo = std::stoi(m[1]);
    std::string after_reg = ops_part.substr(m.position() + m.length());
    size_t offset_pos = after_reg.rfind("0x");
    if (offset_pos == std::string::npos) offset_pos = after_reg.rfind(' ');
    if (offset_pos != std::string::npos) {
      size_t num_start = after_reg.find_last_of(" \t,", after_reg.size()-1);
      if (num_start == std::string::npos) num_start = 0; else num_start++;
      std::string offset_str = after_reg.substr(num_start);
      int64_t offset_val = 0;
      try { offset_val = std::stoll(offset_str, nullptr, 0); } catch (...) {}
      std::string base_part = after_reg.substr(0, num_start);
      size_t be = base_part.find_last_not_of(" \t,");
      if (be != std::string::npos) base_part = base_part.substr(0, be+1);
      auto hexStr = [](int64_t v) { std::ostringstream o; o << std::hex << v; return o.str(); };
      result.push_back("s_load_dwordx2 s[" + std::to_string(lo) + ":" +
                        std::to_string(lo+1) + "]" + base_part + ", 0x" + hexStr(offset_val));
      result.push_back("s_load_dword s" + std::to_string(lo+2) + base_part +
                        ", 0x" + hexStr(offset_val + 8));
    }
  }
  if (result.empty()) {
    std::string new_line = line;
    size_t mpos = new_line.find("s_load_b96");
    new_line.replace(mpos, 10, "s_load_dwordx3");
    std::smatch m2;
    if (std::regex_search(new_line, m2, reg_range)) {
      int lo2 = std::stoi(m2[1]);
      std::string wider = "s[" + std::to_string(lo2) + ":" + std::to_string(lo2 + 2) + "]";
      new_line = new_line.substr(0, m2.position()) + wider +
                 new_line.substr(m2.position() + m2.length());
    }
    result.push_back(new_line);
  }
  return result;
}

// ── Handler: Bitop Instructions ──────────────────────────────────────────────

static TranslationResult HandleBitopInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  if (mnemonic.find("v_bitop") != 0) return std::nullopt;
  std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
  size_t bitop_pos = ops.find("bitop3:");
  if (bitop_pos != std::string::npos) {
    ops = ops.substr(0, bitop_pos);
  }
  auto operands = ParseOperandList(line, mnemonic);
  if (bitop_pos != std::string::npos && operands.size() >= 3) {
    // Remove any bitop3: operands
    std::vector<std::string> clean;
    for (auto& op : operands)
      if (op.find("bitop") == std::string::npos) clean.push_back(op);
    if (clean.size() >= 3) {
      int truth_table_val = 0;
      std::string hex_str2 = ops.substr(ops.find("bitop3:") != std::string::npos ? ops.find("bitop3:") + 7 : 0);
      // Reparse from original
      std::string all_ops = line.substr(line.find(mnemonic) + mnemonic.size());
      size_t bp = all_ops.find("bitop3:");
      if (bp != std::string::npos) {
        try { truth_table_val = std::stoi(all_ops.substr(bp + 7), nullptr, 0); } catch (...) {}
      }
      if (truth_table_val == 0xCA)
        return std::vector<std::string>{"v_bfi_b32 " + clean[0] + ", " + clean[1] + ", " + clean[2] + ", " + clean[0]};
      return std::vector<std::string>{"v_and_b32 " + clean[0] + ", " + clean[1] + ", " + clean[2]};
    }
  }
  if (operands.size() >= 3)
    return std::vector<std::string>{"v_and_b32_e32 " + operands[0] + ", " + operands[1] + ", " + operands[2]};
  return std::vector<std::string>{"s_nop 0 ; UNSUPPORTED: " + line};
}

// ── Handler: SALU Float / Scalar Emulation ───────────────────────────────────

static TranslationResult HandleSALUFloat(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  // s_add_nc_u64 / s_sub_nc_u64
  if (mnemonic == "s_add_nc_u64" || mnemonic == "s_sub_nc_u64") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 3) {
      auto parseSpair = [](const std::string& s) -> std::pair<int,int> {
        auto bracket = s.find('[');
        if (bracket == std::string::npos) return {-1,-1};
        int lo=0, hi=0; size_t p = bracket+1;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') lo=lo*10+(s[p++]-'0');
        if (p<s.size()&&s[p]==':') p++;
        while (p<s.size()&&s[p]>='0'&&s[p]<='9') hi=hi*10+(s[p++]-'0');
        return {lo, hi};
      };
      auto [d0,d1] = parseSpair(operands[0]);
      auto [a0,a1] = parseSpair(operands[1]);
      auto [s0,s1] = parseSpair(operands[2]);
      std::string src_lo = (s0>=0) ? "s"+std::to_string(s0) : operands[2];
      std::string src_hi = (s0>=0) ? "s"+std::to_string(s1) : "0";
      bool is_sub = mnemonic.find("sub") != std::string::npos;
      std::string op = is_sub ? "s_sub_u32" : "s_add_u32";
      std::string opc = is_sub ? "s_subb_u32" : "s_addc_u32";
      return std::vector<std::string>{
        op + " s" + std::to_string(d0) + ", s" + std::to_string(a0) + ", " + src_lo,
        opc + " s" + std::to_string(d1) + ", s" + std::to_string(a1) + ", " + src_hi
      };
    }
    return std::vector<std::string>{"s_nop 0 ; UNSUPPORTED: " + line};
  }

  // s_lshlN_add_u32
  {
    int shift_amt = -1;
    if (mnemonic == "s_lshl1_add_u32") shift_amt = 1;
    else if (mnemonic == "s_lshl2_add_u32") shift_amt = 2;
    else if (mnemonic == "s_lshl3_add_u32") shift_amt = 3;
    else if (mnemonic == "s_lshl4_add_u32") shift_amt = 4;
    if (shift_amt >= 0) {
      auto operands = ParseOperandList(line, mnemonic);
      if (operands.size() >= 3) {
        std::vector<std::string> result;
        result.push_back("s_lshl_b32 " + operands[0] + ", " + operands[1] + ", " + std::to_string(shift_amt));
        if (operands[2] != "0")
          result.push_back("s_add_u32 " + operands[0] + ", " + operands[0] + ", " + operands[2]);
        return result;
      }
    }
  }

  // SALU float → VALU emulation
  {
    static const std::unordered_map<std::string, std::string> kSaluFloatMap = {
        {"s_add_f32", "v_add_f32_e32"}, {"s_sub_f32", "v_sub_f32_e32"},
        {"s_mul_f32", "v_mul_f32_e32"}, {"s_min_f32", "v_min_f32_e32"},
        {"s_max_f32", "v_max_f32_e32"}, {"s_fmac_f32", "v_fmac_f32_e32"},
        {"s_add_f16", "v_add_f16_e32"}, {"s_sub_f16", "v_sub_f16_e32"},
        {"s_mul_f16", "v_mul_f16_e32"}, {"s_min_f16", "v_min_f16_e32"},
        {"s_max_f16", "v_max_f16_e32"},
    };
    auto salu_it = kSaluFloatMap.find(mnemonic);
    if (salu_it != kSaluFloatMap.end()) {
      auto operands = ParseOperandList(line, mnemonic);
      if (operands.size() >= 3) {
        std::vector<std::string> result;
        if (mnemonic == "s_fmac_f32") {
          result.push_back("v_mov_b32_e32 v6, " + operands[1]);
          result.push_back("v_mul_f32_e32 v6, " + operands[2] + ", v6");
          result.push_back("v_add_f32_e32 v6, " + operands[0] + ", v6");
          result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        } else {
          result.push_back("v_mov_b32_e32 v6, " + operands[1]);
          result.push_back(salu_it->second + " v6, " + operands[2] + ", v6");
          result.push_back("v_readfirstlane_b32 " + operands[0] + ", v6");
        }
        return result;
      }
    }
  }

  // s_cvt_*
  if (mnemonic == "s_cvt_f32_f16" || mnemonic == "s_cvt_f16_f32" ||
      mnemonic == "s_cvt_pk_rtz_f16_f32" ||
      mnemonic == "s_cvt_f32_u32" || mnemonic == "s_cvt_f32_i32" ||
      mnemonic == "s_cvt_u32_f32" || mnemonic == "s_cvt_i32_f32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 2) {
      std::string valu_mnem;
      if (mnemonic == "s_cvt_f32_f16") valu_mnem = "v_cvt_f32_f16_e32";
      else if (mnemonic == "s_cvt_f16_f32") valu_mnem = "v_cvt_f16_f32_e32";
      else if (mnemonic == "s_cvt_f32_u32") valu_mnem = "v_cvt_f32_u32_e32";
      else if (mnemonic == "s_cvt_f32_i32") valu_mnem = "v_cvt_f32_i32_e32";
      else if (mnemonic == "s_cvt_u32_f32") valu_mnem = "v_cvt_u32_f32_e32";
      else if (mnemonic == "s_cvt_i32_f32") valu_mnem = "v_cvt_i32_f32_e32";
      else valu_mnem = "v_cvt_pkrtz_f16_f32";
      return std::vector<std::string>{
        valu_mnem + " v6, " + operands[1],
        "v_readfirstlane_b32 " + operands[0] + ", v6"
      };
    }
  }

  // v_s_sqrt_f32
  if (mnemonic == "v_s_sqrt_f32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 2)
      return std::vector<std::string>{
        "v_sqrt_f32_e32 v6, " + operands[1],
        "v_readfirstlane_b32 " + operands[0] + ", v6"
      };
  }

  // s_cmp_*_f32
  {
    static const std::unordered_map<std::string, std::string> kScmpFloatMap = {
      {"s_cmp_gt_f32", "v_cmp_gt_f32_e32"}, {"s_cmp_ge_f32", "v_cmp_ge_f32_e32"},
      {"s_cmp_lt_f32", "v_cmp_lt_f32_e32"}, {"s_cmp_le_f32", "v_cmp_le_f32_e32"},
      {"s_cmp_eq_f32", "v_cmp_eq_f32_e32"}, {"s_cmp_lg_f32", "v_cmp_lg_f32_e32"},
    };
    auto cmp_it = kScmpFloatMap.find(mnemonic);
    if (cmp_it != kScmpFloatMap.end()) {
      auto operands = ParseOperandList(line, mnemonic);
      if (operands.size() >= 2)
        return std::vector<std::string>{
          "v_mov_b32_e32 v6, " + operands[1],
          cmp_it->second + " " + operands[0] + ", v6",
          "s_cmp_lg_u32 vcc_lo, 0"
        };
    }
  }

  return std::nullopt;
}

// ── Handler: VOPD (dual-issue) ───────────────────────────────────────────────

static TranslationResult HandleVOPDInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string& source_cpu, const std::string& target_cpu,
    int scale_temp_vgpr, int cmpx_temp_sgpr, bool compact_mode) {
  if (mnemonic.find("v_dual_") != 0) return std::nullopt;
  size_t sep = line.find("::");
  if (sep == std::string::npos) return std::nullopt;
  std::string first_half = line.substr(0, sep);
  size_t fs = first_half.find_first_not_of(" \t");
  if (fs != std::string::npos) first_half = first_half.substr(fs);
  size_t fe = first_half.find_last_not_of(" \t");
  if (fe != std::string::npos) first_half = first_half.substr(0, fe + 1);
  if (first_half.find("v_dual_") == 0) first_half = "v_" + first_half.substr(7);
  std::string second_half = line.substr(sep + 2);
  size_t ss = second_half.find_first_not_of(" \t");
  if (ss != std::string::npos) second_half = second_half.substr(ss);
  size_t se = second_half.find_last_not_of(" \t");
  if (se != std::string::npos) second_half = second_half.substr(0, se + 1);
  if (second_half.find("v_dual_") == 0) second_half = "v_" + second_half.substr(7);
  auto r1 = TranslateInstruction(first_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
  auto r2 = TranslateInstruction(second_half, source_cpu, target_cpu, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
  std::vector<std::string> result;
  result.insert(result.end(), r1.begin(), r1.end());
  result.insert(result.end(), r2.begin(), r2.end());
  return result;
}

// ── Handler: WMMA Instructions ───────────────────────────────────────────────

static TranslationResult HandleWMMAInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string& target_cpu, int, int, bool);

// ── Handler: Constant Bus Fix ────────────────────────────────────────────────

static TranslationResult HandleConstantBusFix(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool);

// ── Handler: Memory Instructions ─────────────────────────────────────────────

static TranslationResult HandleMemoryInstruction(
    std::string& line, std::string& mnemonic,
    const std::string&, const std::string&, int scale_temp_vgpr, int, bool);

// ── Handler: EXEC Operations ─────────────────────────────────────────────────

static TranslationResult HandleExecOperation(
    std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&,
    int scale_temp_vgpr, int cmpx_temp_sgpr, bool compact_mode);


// ── Handler: v_add_nc_u64 / v_mad_u32 ────────────────────────────────────────

static TranslationResult Handle64BitVALU(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  if (mnemonic == "v_add_nc_u64" || mnemonic == "v_add_nc_u64_e32" ||
      mnemonic == "v_add_u64" || mnemonic == "v_add_u64_e32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    struct RegOrImm { char prefix; int lo; int hi; std::string imm; };
    auto parseOperand = [](const std::string& s, size_t& pos) -> RegOrImm {
      while (pos < s.size() && (s[pos]==' '||s[pos]==','||s[pos]=='\t')) ++pos;
      if (pos >= s.size()) return {'?', -1, -1, ""};
      char prefix = s[pos];
      if (prefix == 'v' || prefix == 's') {
        ++pos;
        if (pos < s.size() && s[pos] == '[') {
          ++pos; int lo=0, hi=0;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') lo=lo*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==':') ++pos;
          while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') hi=hi*10+(s[pos++]-'0');
          if (pos<s.size()&&s[pos]==']') ++pos;
          return {prefix, lo, hi, ""};
        }
        return {'?', -1, -1, ""};
      }
      size_t st = pos;
      if (s[pos]=='-') ++pos;
      while (pos<s.size()&&(std::isalnum(s[pos])||s[pos]=='x')) ++pos;
      return {'#', 0, 0, s.substr(st, pos-st)};
    };
    size_t pos = 0;
    auto d = parseOperand(ops, pos);
    auto a = parseOperand(ops, pos);
    auto b = parseOperand(ops, pos);
    if (d.lo >= 0 && (a.lo >= 0 || a.prefix == '#') && (b.lo >= 0 || b.prefix == '#')) {
      auto fmtPair = [&](char p, int lo, int hi) -> std::string {
        return std::string(1, p) + "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
      };
      std::vector<std::string> result;
      std::string src0_pair;
      if (a.prefix == 'v' || a.prefix == 's')
        src0_pair = fmtPair(a.prefix, a.lo, a.hi);
      else {
        result.push_back("v_mov_b32_e32 v252, " + a.imm);
        result.push_back("v_mov_b32_e32 v253, 0");
        src0_pair = "v[252:253]";
      }
      std::string src1 = (b.prefix == '#') ? b.imm : fmtPair(b.prefix, b.lo, b.hi);
      result.push_back("v_lshl_add_u64 " + fmtPair(d.prefix, d.lo, d.hi) + ", " + src0_pair + ", 0, " + src1);
      return result;
    }
    return std::vector<std::string>{"s_nop 0 ; UNSUPPORTED: " + line};
  }

  if (mnemonic == "v_mad_u32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 4) {
      std::vector<std::string> result;
      std::string vdst = operands[0], src0 = operands[1], src1 = operands[2], src2 = operands[3];
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      if (vdst != src2) {
        if (src0_sgpr && src1_sgpr) {
          result.push_back("v_mov_b32_e32 v6, " + src0);
          result.push_back("v_mul_lo_u32 " + vdst + ", v6, " + src1);
        } else {
          result.push_back("v_mul_lo_u32 " + vdst + ", " + src0 + ", " + src1);
        }
        result.push_back("v_add_u32_e32 " + vdst + ", " + vdst + ", " + src2);
      } else {
        result.push_back("v_mov_b32_e32 v6, " + src0);
        result.push_back("v_mul_lo_u32 v6, v6, " + src1);
        result.push_back("v_add_u32_e32 " + vdst + ", v6, " + src2);
      }
      return result;
    }
  }

  return std::nullopt;
}

// ── HandleConstantBusFix implementation ──────────────────────────────────────

static TranslationResult HandleConstantBusFix(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&, int, int, bool) {
  // v_cndmask_b32_e64
  if (mnemonic == "v_cndmask_b32_e64") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 4) {
      std::vector<std::string> result;
      for (auto& op : operands) { if (op == "vcc_lo") op = "vcc"; }
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos && mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      std::string& src1 = operands[2];
      bool src1_sgpr = !src1.empty() && src1[0] == 's';
      bool mask_sgpr = !mask.empty() && (mask[0] == 's' || mask.substr(0, 3) == "vcc" || mask.substr(0, 4) == "exec");
      if (src1_sgpr && mask_sgpr) {
        result.push_back("v_mov_b32_e32 v6, " + src1);
        src1 = "v6";
      }
      std::string fixed = mnemonic + " " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      result.push_back(fixed);
      return result;
    }
  }
  // v_cndmask_b32_e32 SGPR src0
  if (mnemonic == "v_cndmask_b32_e32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 3) {
      std::string& src0 = operands[1];
      bool src0_sgpr = !src0.empty() && src0[0] == 's';
      bool src0_lit = src0.size() > 2 && src0[0] == '0' && (src0[1] == 'x' || src0[1] == 'X');
      if (src0_sgpr || src0_lit) {
        std::vector<std::string> result;
        result.push_back("v_mov_b32_e32 v6, " + src0);
        src0 = "v6";
        std::string fixed = mnemonic + " " + operands[0];
        for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
        result.push_back(fixed);
        return result;
      }
    }
  }
  // v_cndmask_b32 (no suffix) with explicit mask
  if (mnemonic == "v_cndmask_b32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 4) {
      for (auto& op : operands) { if (op == "vcc_lo") op = "vcc"; }
      auto& mask = operands[3];
      if (!mask.empty() && mask[0] == 's' && mask.find('[') == std::string::npos &&
          mask.find("vcc") == std::string::npos && mask.find("exec") == std::string::npos &&
          mask.size() > 1 && std::isdigit((unsigned char)mask[1])) {
        int n = std::stoi(mask.substr(1));
        int even = n & ~1;
        mask = "s[" + std::to_string(even) + ":" + std::to_string(even + 1) + "]";
      }
      std::string fixed = "v_cndmask_b32_e64 " + operands[0];
      for (size_t i = 1; i < operands.size(); ++i) fixed += ", " + operands[i];
      return std::vector<std::string>{fixed};
    }
  }
  // DPP8
  if (line.find("dpp8:") != std::string::npos) {
    size_t dpp8_pos = line.find("dpp8:[");
    if (dpp8_pos != std::string::npos) {
      std::string base_part = line.substr(0, dpp8_pos);
      size_t be = base_part.find_last_not_of(" \t");
      if (be != std::string::npos) base_part = base_part.substr(0, be + 1);
      if (line.find("dpp8:[0,1,2,3,4,5,6,7]") != std::string::npos) {
        std::string no_dpp = base_part;
        size_t dpp_suffix = no_dpp.find("_dpp");
        if (dpp_suffix != std::string::npos) no_dpp.replace(dpp_suffix, 4, "_e32");
        return std::vector<std::string>{no_dpp};
      }
      return std::vector<std::string>{base_part + " row_shr:0 row_mask:0xf bank_mask:0xf"};
    }
  }
  // VOP3 with 32-bit literal
  {
    bool is_vop3_candidate =
        mnemonic == "v_fma_f32" || mnemonic == "v_fma_f16" ||
        mnemonic == "v_fma_f64" || mnemonic == "v_ldexp_f32" ||
        mnemonic == "v_div_fmas_f32" || mnemonic == "v_div_fixup_f32" ||
        mnemonic == "v_med3_f32" || mnemonic == "v_med3_i32" ||
        mnemonic == "v_bfi_b32" || mnemonic == "v_alignbit_b32";
    if (is_vop3_candidate) {
      auto operands = ParseOperandList(line, mnemonic);
      if (operands.size() >= 3) {
        for (size_t i = 1; i < operands.size(); ++i) {
          std::string op = operands[i];
          bool neg = !op.empty() && op[0] == '-';
          if (neg) op = op.substr(1);
          if (op.size() > 2 && op[0] == '0' && (op[1] == 'x' || op[1] == 'X')) {
            std::vector<std::string> result;
            result.push_back("v_mov_b32_e32 v6, " + op);
            operands[i] = (neg ? "-" : "") + std::string("v6");
            std::string fixed = mnemonic + " " + operands[0];
            for (size_t j = 1; j < operands.size(); ++j) fixed += ", " + operands[j];
            result.push_back(fixed);
            return result;
          }
        }
      }
    }
  }
  // v_perm_b32 with literal constant
  if (mnemonic == "v_perm_b32") {
    std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
    if (ops.find("0x") != std::string::npos) {
      auto operands = ParseOperandList(line, mnemonic);
      if (operands.size() >= 4) {
        return std::vector<std::string>{
          "s_mov_b32 s13, " + operands[3],
          "v_perm_b32 " + operands[0] + ", " + operands[1] + ", " + operands[2] + ", s13"
        };
      }
    }
  }
  // v_fmamk_f32 / v_fmaak_f32
  if (mnemonic == "v_fmamk_f32" || mnemonic == "v_fmaak_f32") {
    auto operands = ParseOperandList(line, mnemonic);
    if (operands.size() >= 4) {
      std::vector<std::string> result;
      const std::string& literal = (mnemonic == "v_fmamk_f32") ? operands[2] : operands[3];
      result.push_back("v_mov_b32_e32 v6, " + literal);
      if (mnemonic == "v_fmamk_f32")
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", v6, " + operands[3]);
      else
        result.push_back("v_fma_f32 " + operands[0] + ", " + operands[1] + ", " + operands[2] + ", v6");
      return result;
    }
  }
  return std::nullopt;
}

// ── HandleMemoryInstruction implementation ───────────────────────────────────

static TranslationResult HandleMemoryInstruction(
    std::string& line, std::string& mnemonic,
    const std::string&, const std::string&, int scale_temp_vgpr, int, bool) {
  // scale_offset emulation
  if (line.find("scale_offset") != std::string::npos) {
    int shift = 0;
    if (mnemonic.find("_b32") != std::string::npos || mnemonic.find("_dword") != std::string::npos) shift = 2;
    else if (mnemonic.find("_b64") != std::string::npos || mnemonic.find("_dwordx2") != std::string::npos) shift = 3;
    else if (mnemonic.find("_b128") != std::string::npos || mnemonic.find("_dwordx4") != std::string::npos) shift = 4;
    else if (mnemonic.find("_b16") != std::string::npos) shift = 1;
    if (shift > 0) {
      size_t mnem_end = line.find(mnemonic) + mnemonic.size();
      std::string ops = line.substr(mnem_end);
      bool is_store = mnemonic.find("store") != std::string::npos;
      std::string vaddr;
      if (is_store) {
        size_t s = ops.find_first_not_of(" \t");
        size_t e = ops.find_first_of(" \t,", s);
        if (s != std::string::npos) vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
      } else {
        size_t comma1 = ops.find(',');
        if (comma1 != std::string::npos) {
          size_t s = ops.find_first_not_of(" \t,", comma1 + 1);
          size_t e = ops.find_first_of(" \t,", s);
          if (s != std::string::npos) vaddr = ops.substr(s, e != std::string::npos ? e - s : std::string::npos);
        }
      }
      if (!vaddr.empty()) {
        std::vector<std::string> result;
        const std::string kScaleTemp = "v" + std::to_string(scale_temp_vgpr);
        result.push_back("v_lshlrev_b32_e32 " + kScaleTemp + ", " + std::to_string(shift) + ", " + vaddr);
        std::string modified = line;
        size_t vaddr_pos = modified.find(vaddr, mnem_end);
        if (vaddr_pos != std::string::npos) modified.replace(vaddr_pos, vaddr.size(), kScaleTemp);
        size_t so_pos = modified.find("scale_offset");
        if (so_pos != std::string::npos) {
          if (so_pos > 0 && modified[so_pos-1] == ' ') --so_pos;
          modified.erase(so_pos);
        }
        line = modified;
        mnemonic = TranspileExtractMnemonic(line);
      }
    }
  }
  return std::nullopt;
}

// ── HandleWMMAInstruction implementation ─────────────────────────────────────
// (The original WMMA→MFMA logic is complex; extracted as-is from TranslateInstruction)

static TranslationResult HandleWMMAInstruction(
    const std::string& line, const std::string& mnemonic,
    const std::string&, const std::string& target_cpu, int, int, bool) {
  struct WmmaMfmaMapping { const char* wmma_mnem; const char* mfma_mnem; int dst_vgprs_w32; int src_vgprs_w32; int dst_vgprs_w64; int src_vgprs_w64; };
  static const WmmaMfmaMapping kWmmaMap[] = {
    {"v_wmma_f32_16x16x32_f16", "v_mfma_f32_16x16x32_f16", 8, 8, 4, 4},
    {"v_wmma_f32_16x16x32_bf16", "v_mfma_f32_16x16x32bf16", 8, 8, 4, 4},
    {"v_wmma_f32_16x16x16_f16", "v_mfma_f32_16x16x16_f16", 8, 4, 4, 2},
    {"v_wmma_f32_16x16x4_f32", "v_mfma_f32_16x16x4_f32", 8, 2, 4, 1},
    {"v_wmma_i32_16x16x64_iu8", "v_mfma_i32_16x16x64_i8", 8, 8, 4, 4},
  };
  const WmmaMfmaMapping* mapping = nullptr;
  for (const auto& m : kWmmaMap)
    if (mnemonic == m.wmma_mnem) { mapping = &m; break; }
  if (!mapping) {
    if (mnemonic.find("v_wmma_") == 0 || mnemonic.find("v_swmmac_") == 0)
      return std::vector<std::string>{"s_nop 0 ; UNSUPPORTED WMMA: " + mnemonic};
    return std::nullopt;
  }
  std::string ops = line.substr(line.find(mnemonic) + mnemonic.size());
  size_t op_start = ops.find_first_not_of(" \t");
  if (op_start != std::string::npos) ops = ops.substr(op_start);
  auto parseVRegRange = [](const std::string& s, size_t& pos) -> std::pair<int, int> {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\t')) ++pos;
    if (pos >= s.size() || s[pos] != 'v') return {-1, -1};
    ++pos;
    if (pos < s.size() && s[pos] == '[') {
      ++pos; int lo = 0, hi = 0;
      while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') lo = lo * 10 + (s[pos++] - '0');
      if (pos < s.size() && s[pos] == ':') ++pos;
      while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') hi = hi * 10 + (s[pos++] - '0');
      if (pos < s.size() && s[pos] == ']') ++pos;
      return {lo, hi};
    }
    return {-1, -1};
  };
  size_t pos = 0;
  auto dst = parseVRegRange(ops, pos);
  auto srcA = parseVRegRange(ops, pos);
  auto srcB = parseVRegRange(ops, pos);
  auto acc = parseVRegRange(ops, pos);
  if (dst.first < 0 || srcA.first < 0 || srcB.first < 0 || acc.first < 0)
    return std::vector<std::string>{"s_nop 0 ; WMMA parse failed: " + line};
  int src_w64 = mapping->src_vgprs_w64;
  int dst_w64 = mapping->dst_vgprs_w64;
  std::string mfma_mnem = mapping->mfma_mnem;
  int t_lane = 248, t_src = 249, t_addr = 250, t_upper = 251, t0 = 252, t1 = 253;
  int mfma_srcA = srcA.first, mfma_srcB = srcB.first, mfma_acc = acc.first, mfma_dst = dst.first;
  auto regRange = [](int base, int count) -> std::string {
    if (count == 1) return "v" + std::to_string(base);
    return "v[" + std::to_string(base) + ":" + std::to_string(base + count - 1) + "]";
  };
  std::vector<std::string> result;
  auto emitRedistribute = [&](int base_w32, int out_base, int n_w64) {
    for (int i = 0; i < n_w64; ++i) {
      int lo_reg = base_w32 + i, hi_reg = base_w32 + n_w64 + i;
      result.push_back("ds_bpermute_b32 v" + std::to_string(t0) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(lo_reg));
      result.push_back("ds_bpermute_b32 v" + std::to_string(t1) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(hi_reg));
      result.push_back("s_waitcnt lgkmcnt(0)");
      result.push_back("v_cndmask_b32_e32 v" + std::to_string(out_base + i) + ", v" + std::to_string(t0) + ", v" + std::to_string(t1) + ", vcc");
    }
  };
  result.push_back("; BEGIN WMMA→MFMA: " + mnemonic + " → " + mfma_mnem);
  result.push_back("s_mov_b32 s12, exec_lo");
  result.push_back("s_mov_b32 s13, exec_hi");
  result.push_back("s_mov_b64 exec, -1");
  result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
  result.push_back("v_mbcnt_hi_u32_b32 v" + std::to_string(t_lane) + ", -1, v" + std::to_string(t_lane));
  result.push_back("v_lshrrev_b32_e32 v" + std::to_string(t_src) + ", 1, v" + std::to_string(t_lane));
  result.push_back("v_and_b32_e32 v" + std::to_string(t_upper) + ", 1, v" + std::to_string(t_lane));
  result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) + ", 2, v" + std::to_string(t_src));
  result.push_back("v_cmp_ne_u32 vcc, 0, v" + std::to_string(t_upper));
  emitRedistribute(srcA.first, mfma_srcA, src_w64);
  emitRedistribute(srcB.first, mfma_srcB, src_w64);
  emitRedistribute(acc.first, mfma_acc, dst_w64);
  bool need_decompose = false;
  std::string actual_mfma = mfma_mnem;
  if (target_cpu.find("gfx942") != std::string::npos || target_cpu.find("gfx940") != std::string::npos || target_cpu.find("gfx941") != std::string::npos) {
    if (mfma_mnem == "v_mfma_f32_16x16x32_f16" || mfma_mnem == "v_mfma_f32_16x16x32bf16") {
      need_decompose = true;
      actual_mfma = (mfma_mnem.find("bf16") != std::string::npos) ? "v_mfma_f32_16x16x16bf16_1k" : "v_mfma_f32_16x16x16_f16";
    }
  }
  if (need_decompose) {
    int half_src = src_w64 / 2;
    result.push_back(actual_mfma + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA, half_src) + ", " + regRange(mfma_srcB, half_src) + ", " + regRange(mfma_acc, dst_w64));
    result.push_back(actual_mfma + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA + half_src, half_src) + ", " + regRange(mfma_srcB + half_src, half_src) + ", " + regRange(mfma_dst, dst_w64));
  } else {
    result.push_back(mfma_mnem + " " + regRange(mfma_dst, dst_w64) + ", " + regRange(mfma_srcA, src_w64) + ", " + regRange(mfma_srcB, src_w64) + ", " + regRange(mfma_acc, dst_w64));
  }
  result.push_back("s_mov_b32 exec_lo, s12");
  result.push_back("s_mov_b32 exec_hi, 0");
  result.push_back("v_mbcnt_lo_u32_b32 v" + std::to_string(t_lane) + ", -1, 0");
  result.push_back("v_lshlrev_b32_e32 v" + std::to_string(t_addr) + ", 3, v" + std::to_string(t_lane));
  result.push_back("v_add_u32_e32 v" + std::to_string(t0) + ", 4, v" + std::to_string(t_addr));
  result.push_back("s_mov_b64 exec, -1");
  for (int i = 0; i < dst_w64; ++i) {
    result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + i) + ", v" + std::to_string(t_addr) + ", v" + std::to_string(mfma_dst + i));
    result.push_back("ds_bpermute_b32 v" + std::to_string(dst.first + dst_w64 + i) + ", v" + std::to_string(t0) + ", v" + std::to_string(mfma_dst + i));
  }
  result.push_back("s_waitcnt lgkmcnt(0)");
  result.push_back("s_mov_b32 exec_lo, s12");
  result.push_back("s_mov_b32 exec_hi, s13");
  result.push_back("; END WMMA→MFMA: " + mfma_mnem);
  return result;
}

// ── HandleExecOperation implementation ───────────────────────────────────────
// Handles v_cmpx, v_cmp with SGPR dest, VCC/EXEC widening (the "tail" of the
// original TranslateInstruction that runs after mnemonic renaming).

static TranslationResult HandleExecOperation(
    std::string& line, const std::string& mnemonic,
    const std::string&, const std::string&,
    int scale_temp_vgpr, int cmpx_temp_sgpr, bool compact_mode) {
  (void)scale_temp_vgpr;
  (void)cmpx_temp_sgpr;
  (void)compact_mode;
  // v_div_scale_f32 null sdst → vcc
  if (mnemonic == "v_div_scale_f32") {
    size_t null_pos = line.find(", null,");
    if (null_pos != std::string::npos)
      line.replace(null_pos, 7, ", vcc,");
  }
  return std::nullopt;
}


// ── Helper: v_cmp / v_cmpx wave64 expansion ──────────────────────────────────

static bool HandleVCmpExpansion(
    std::string& line, const std::string& mnemonic,
    std::vector<std::string>& result,
    int scale_temp_vgpr, int cmpx_temp_sgpr, bool compact_mode) {
  // v_cmp_*_e64 with vcc dest + large literal → VOPC _e32
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops_str = line.substr(op_start);
    size_t sp = ops_str.find_first_not_of(" \t");
    if (sp != std::string::npos) ops_str = ops_str.substr(sp);
    size_t first_comma = ops_str.find(',');
    if (first_comma != std::string::npos) {
      std::string first_op = ops_str.substr(0, first_comma);
      size_t fe = first_op.find_last_not_of(" \t");
      if (fe != std::string::npos) first_op = first_op.substr(0, fe + 1);
      if (first_op == "vcc" || first_op == "vcc_lo") {
        auto srcs = ParseOperandList(ops_str.substr(first_comma + 1), "");
        if (srcs.size() >= 2) {
          auto is_ll = [](const std::string& s) -> bool {
            if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return true;
            try {
              if (!s.empty() && std::isdigit((unsigned char)s[0]) && std::stol(s) > 64) return true;
              if (s.size() > 1 && s[0] == '-' && std::isdigit((unsigned char)s[1]) && std::stol(s) < -16) return true;
            } catch (...) {}
            return false;
          };
          std::string base_mnem = mnemonic.substr(0, mnemonic.size() - 4) + "_e32";
          if (is_ll(srcs[1]) && !is_ll(srcs[0])) {
            result.push_back("v_mov_b32_e32 v6, " + srcs[1]);
            result.push_back(base_mnem + " " + srcs[0] + ", v6");
            return true;
          } else if (is_ll(srcs[0])) {
            result.push_back(base_mnem + " " + srcs[0] + ", " + srcs[1]);
            return true;
          }
        }
      }
    }
  }

  // v_cmpx _e64
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    std::string base_mnem = mnemonic.substr(0, mnemonic.find("_e64"));
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s = ops.find_first_not_of(" \t");
    if (s != std::string::npos) ops = ops.substr(s);
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(base_mnem + " " + ops);
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return true;
  }

  // v_cmpx _e32
  if (mnemonic.find("v_cmpx_") == 0 && mnemonic.find("_e64") == std::string::npos) {
    std::string stemp = "s" + std::to_string(cmpx_temp_sgpr);
    result.push_back("s_mov_b32 " + stemp + ", vcc_lo");
    result.push_back(line);
    result.push_back("s_mov_b32 exec_hi, 0");
    result.push_back("s_mov_b32 vcc_lo, " + stemp);
    return true;
  }

  // v_cmp _e64 with SGPR dest → expand to SGPR pair
  if (mnemonic.find("v_cmp_") == 0 && mnemonic.find("_e64") != std::string::npos) {
    size_t op_start = line.find(mnemonic) + mnemonic.size();
    std::string ops = line.substr(op_start);
    size_t s_pos = ops.find_first_not_of(" \t");
    if (s_pos != std::string::npos) {
      std::string trimmed = ops.substr(s_pos);
      if (!trimmed.empty() && trimmed[0] == 's' && trimmed.size() > 1 && std::isdigit(trimmed[1])) {
        size_t comma = trimmed.find(',');
        if (comma != std::string::npos) {
          std::string dst_str = trimmed.substr(0, comma);
          size_t de = dst_str.find_last_not_of(" \t");
          dst_str = dst_str.substr(0, de + 1);
          int reg_num = std::stoi(dst_str.substr(1));
          int even = reg_num & ~1;
          int odd = even + 1;
          std::string pair = "s[" + std::to_string(even) + ":" + std::to_string(odd) + "]";
          std::string rest = trimmed.substr(comma);
          std::string save_reg = "v" + std::to_string(scale_temp_vgpr);
          line = mnemonic + " " + pair + rest;
          if (reg_num == even) {
            if (compact_mode) {
              result.push_back(line);
            } else {
              result.push_back("v_mov_b32_e32 " + save_reg + ", s" + std::to_string(odd));
              result.push_back(line);
              result.push_back("v_readfirstlane_b32 s" + std::to_string(odd) + ", " + save_reg);
            }
          } else {
            result.push_back("v_mov_b32_e32 " + save_reg + ", s" + std::to_string(even));
            result.push_back(line);
            result.push_back("s_mov_b32 s" + std::to_string(reg_num) + ", s" + std::to_string(even));
            result.push_back("v_readfirstlane_b32 s" + std::to_string(even) + ", " + save_reg);
          }
          auto is_ll = [](const std::string& sv) -> bool {
            if (sv.empty()) return false;
            if (sv.size() > 2 && sv[0] == '0' && (sv[1] == 'x' || sv[1] == 'X')) return true;
            try {
              if (std::isdigit((unsigned char)sv[0]) && std::stol(sv) > 64) return true;
              if (sv[0] == '-' && sv.size() > 1 && std::stol(sv) < -16) return true;
            } catch (...) {}
            return false;
          };
          for (size_t ri = 0; ri < result.size(); ri++) {
            if (result[ri].find("v_cmp_") != std::string::npos) {
              size_t lc = result[ri].rfind(',');
              if (lc != std::string::npos) {
                std::string lo = result[ri].substr(lc + 1);
                size_t ls = lo.find_first_not_of(" \t");
                size_t le = lo.find_last_not_of(" \t");
                if (ls != std::string::npos) {
                  std::string lt = lo.substr(ls, le - ls + 1);
                  if (is_ll(lt)) {
                    result.insert(result.begin() + ri, "v_mov_b32_e32 v6, " + lt);
                    result[ri + 1] = result[ri + 1].substr(0, lc + 1) + " v6";
                  }
                }
              }
              break;
            }
          }
          return true;
        }
      }
    }
    auto is_ll = [](const std::string& sv) -> bool {
      if (sv.empty()) return false;
      if (sv.size() > 2 && sv[0] == '0' && (sv[1] == 'x' || sv[1] == 'X')) return true;
      try {
        if (std::isdigit((unsigned char)sv[0]) && std::stol(sv) > 64) return true;
        if (sv[0] == '-' && sv.size() > 1 && std::stol(sv) < -16) return true;
      } catch (...) {}
      return false;
    };
    size_t last_comma = line.rfind(',');
    if (last_comma != std::string::npos) {
      std::string last_op = line.substr(last_comma + 1);
      size_t ls = last_op.find_first_not_of(" \t");
      size_t le = last_op.find_last_not_of(" \t");
      if (ls != std::string::npos) {
        std::string lt = last_op.substr(ls, le - ls + 1);
        if (is_ll(lt)) {
          result.push_back("v_mov_b32_e32 v6, " + lt);
          line = line.substr(0, last_comma + 1) + " v6";
        }
      }
    }
  }

  return false;
}

// ── TranslateInstruction — Table-driven dispatch ─────────────────────────────

static std::vector<std::string> TranslateInstruction(const std::string& asm_line,
                                               const std::string& source_cpu,
                                               const std::string& target_cpu,
                                               int scale_temp_vgpr = 7,
                                               int cmpx_temp_sgpr = 16,
                                               bool compact_mode = false) {
  std::vector<std::string> result;
  std::string line = asm_line;

  // ── Pre-processing: strip whitespace and comments ──
  size_t start = line.find_first_not_of(" \t");
  if (start == std::string::npos) { result.push_back(line); return result; }
  if (start > 0) line = line.substr(start);

  size_t comment = line.find("//");
  if (comment != std::string::npos) {
    line = line.substr(0, comment);
    size_t end = line.find_last_not_of(" \t");
    if (end != std::string::npos) line = line.substr(0, end + 1);
  }
  if (line.empty()) { result.push_back(""); return result; }

  std::string mnemonic = TranspileExtractMnemonic(line);

  // GFX12 _nc_ VALU → remove _nc_
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed = mnemonic;
    size_t nc_pos = fixed.find("_nc_");
    fixed.replace(nc_pos, 4, "_");
    if (fixed.find("_e32") == std::string::npos && fixed.find("_e64") == std::string::npos)
      fixed += "_e32";
    line = TranspileReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // GFX12 s_and_not1/s_or_not1 → s_andn2/s_orn2
  if (mnemonic.find("_not1_") != std::string::npos && mnemonic[0] == 's') {
    std::string fixed = mnemonic;
    size_t not1_pos = fixed.find("_not1_");
    fixed.replace(not1_pos, 6, "n2_");
    line = TranspileReplaceMnemonic(line, mnemonic, fixed);
    mnemonic = fixed;
  }

  // ── Dispatch: try each category handler ──

  using HandlerFn = TranslationResult (*)(
      const std::string&, const std::string&, const std::string&,
      const std::string&, int, int, bool);

  static const struct { const char* name; HandlerFn handler; } kHandlers[] = {
    {"bitop",         HandleBitopInstruction},
    {"wait",          HandleWaitInstruction},
    {"unsupported",   HandleUnsupportedInstruction},
    {"smem",          HandleSMEMInstruction},
    {"salu_float",    HandleSALUFloat},
    {"64bit_valu",    Handle64BitVALU},
    {"constant_bus",  HandleConstantBusFix},
    {"vopd",          HandleVOPDInstruction},
    {"wmma",          HandleWMMAInstruction},
  };

  for (const auto& h : kHandlers) {
    auto r = h.handler(line, mnemonic, source_cpu, target_cpu,
                       scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
    if (r.has_value()) return *r;
  }

  // Memory handler may modify line/mnemonic (scale_offset)
  {
    auto r = HandleMemoryInstruction(line, mnemonic, source_cpu, target_cpu,
                                     scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
    if (r.has_value()) return *r;
  }

  // ExecOperation handler (v_div_scale null→vcc side effect)
  {
    auto r = HandleExecOperation(line, mnemonic, source_cpu, target_cpu,
                                 scale_temp_vgpr, cmpx_temp_sgpr, compact_mode);
    if (r.has_value()) return *r;
  }

  // ── Post-dispatch: common transforms ──

  // Remaining _nc_ cleanup
  if (mnemonic.find("_nc_") != std::string::npos && mnemonic[0] == 'v') {
    std::string fixed_mnem = mnemonic;
    size_t nc_pos = fixed_mnem.find("_nc_");
    fixed_mnem.replace(nc_pos, 4, "_");
    line = TranspileReplaceMnemonic(line, mnemonic, fixed_mnem);
    mnemonic = fixed_mnem;
  }

  // Unsupported final check
  if (IsUnsupportedOnGFX9(mnemonic)) {
    result.push_back("s_nop 0 ; UNSUPPORTED: " + mnemonic);
    return result;
  }

  // Flat+saddr → Global conversion
  {
    bool is_flat_with_saddr = false;
    if (mnemonic.find("flat_load_") == 0 || mnemonic.find("flat_store_") == 0) {
      size_t s_bracket = line.find("s[", line.find(mnemonic) + mnemonic.size());
      if (s_bracket != std::string::npos) is_flat_with_saddr = true;
    }
    if (is_flat_with_saddr) {
      size_t flat_pos = mnemonic.find("flat_");
      if (flat_pos != std::string::npos) {
        std::string global_mnemonic = "global_" + mnemonic.substr(flat_pos + 5);
        line = TranspileReplaceMnemonic(line, mnemonic, global_mnemonic);
        mnemonic = global_mnemonic;
      }
    }
  }

  // Mnemonic renaming via table
  const auto& mnem_map = GetMnemonicMap();
  auto it = mnem_map.find(mnemonic);
  if (it != mnem_map.end()) {
    line = TranspileReplaceMnemonic(line, mnemonic, it->second);
    mnemonic = it->second;
  } else {
    std::string base = mnemonic;
    std::string suffix;
    if (base.size() > 4 && base.substr(base.size() - 4) == "_e32") { suffix = "_e32"; base = base.substr(0, base.size() - 4); }
    else if (base.size() > 4 && base.substr(base.size() - 4) == "_e64") { suffix = "_e64"; base = base.substr(0, base.size() - 4); }
    if (!suffix.empty()) {
      it = mnem_map.find(base);
      if (it != mnem_map.end()) {
        std::string new_mnem = it->second + suffix;
        line = TranspileReplaceMnemonic(line, mnemonic, new_mnem);
        mnemonic = new_mnem;
      }
    }
  }

  // Operand syntax translation
  line = TranslateOperandSyntax(line, mnemonic);

  // v_cmp / v_cmpx wave64 expansion
  if (HandleVCmpExpansion(line, mnemonic, result, scale_temp_vgpr, cmpx_temp_sgpr, compact_mode))
    return result;

  // VCC width translation
  bool is_b32_scalar = (mnemonic.find("_b32") != std::string::npos && mnemonic[0] == 's');
  if (!is_b32_scalar)
    line = WidenVccReferences(line);
  if (is_b32_scalar) {
    size_t mnem_end = line.find_first_of(" \t");
    if (mnem_end != std::string::npos) {
      std::string ops_part = line.substr(mnem_end);
      size_t pos = 0;
      while ((pos = ops_part.find("vcc", pos)) != std::string::npos) {
        size_t end = pos + 3;
        if (end < ops_part.size() && ops_part[end] == '_') { pos = end; continue; }
        ops_part.replace(pos, 3, "vcc_lo");
        pos += 6;
      }
      line = line.substr(0, mnem_end) + ops_part;
    }
  }

  // EXEC width adaptation (wave32 → wave64)
  auto exec_result = WidenExecOperation(line, compact_mode);
  for (auto& l : exec_result)
    result.push_back(std::move(l));
  return result;
}



// ── TranspileStats ───────────────────────────────────────────────────────────

struct TranspileStats {
  uint32_t total_instructions = 0;
  uint32_t translated_passthrough = 0;
  uint32_t translated_renamed = 0;
  uint32_t translated_waitcnt = 0;
  uint32_t translated_exec = 0;
  uint32_t unsupported_skipped = 0;
};

// ── TranspileCodeObject ──────────────────────────────────────────────────────

static amd_comgr_status_t
TranspileCodeObject(const void *elf_data, size_t elf_size,
                    const std::string &source_isa,
                    const std::string &target_isa,
                    void **out_data, size_t *out_size,
                    amd_comgr_hotswap_result_t *result) {
  TranspileStats stats;
  std::string src_cpu = ExtractCPU(source_isa);
  std::string tgt_cpu = ExtractCPU(target_isa);

  std::cerr << "hotswap: transpile: " << src_cpu << " → " << tgt_cpu << "\n";

  const uint8_t* elf = static_cast<const uint8_t*>(elf_data);
  size_t size = elf_size;

  ElfInfo elf_info;
  if (!ParseElfInfo(elf, size, elf_info)) {
    std::cerr << "hotswap: transpile: failed to parse ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  if (elf_info.text_size == 0) {
    std::cerr << "hotswap: transpile: empty .text section\n";
    uint8_t *buf = static_cast<uint8_t *>(std::malloc(elf_size));
    if (!buf) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(buf, elf_data, elf_size);
    *out_data = buf;
    *out_size = elf_size;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  LLVMState &src_state = InitLLVMCached(source_isa);
  if (!src_state.valid) {
    std::cerr << "hotswap: transpile: failed to init source ISA '" << source_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  const uint8_t* text = elf + elf_info.text_offset;

  struct KernelInfo {
    uint64_t desc_offset;
    uint64_t code_offset;
  };
  std::vector<KernelInfo> kernels;
  for (uint64_t off = 0; off + 256 <= elf_info.text_size; off += 256) {
    uint64_t entry_offset;
    std::memcpy(&entry_offset, text + off + 16, 8);
    if (entry_offset == 256)
      kernels.push_back({off, off + 256});
  }

  if (kernels.empty()) {
    uint64_t code_offset_in_text = 0;
    uint64_t text_vaddr = 0;
    if (elf_info.text_idx >= 0)
      text_vaddr = elf_info.sections[elf_info.text_idx].addr;
    for (const auto& sec : elf_info.sections) {
      if (sec.name == ".rodata" && sec.size >= 64) {
        for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
          const uint8_t* desc = elf + sec.offset + off;
          uint64_t entry;
          std::memcpy(&entry, desc + 16, 8);
          if (entry > 0 && entry < 1000000) {
            uint64_t kd_vaddr = sec.addr + off;
            uint64_t code_vaddr = kd_vaddr + entry;
            if (code_vaddr >= text_vaddr)
              code_offset_in_text = code_vaddr - text_vaddr;
            break;
          }
        }
        break;
      }
    }
    std::cerr << "hotswap: transpile: no embedded descriptors in .text, "
              << "code at .text internal offset " << code_offset_in_text << "\n";
    kernels.push_back({code_offset_in_text, code_offset_in_text});
  } else {
    std::cerr << "hotswap: transpile: found " << kernels.size()
              << " embedded kernel descriptor(s)\n";
  }

  std::string translated_asm;
  translated_asm += ".text\n";

  for (size_t ki = 0; ki < kernels.size(); ++ki) {
    auto& kern = kernels[ki];

    uint64_t emit_end = kern.code_offset;
    uint64_t emit_start = (kern.desc_offset != kern.code_offset) ? kern.desc_offset : 0;
    for (uint64_t i = emit_start; i < emit_end; i += 4) {
      if (i + 4 > elf_info.text_size) break;
      uint32_t word;
      std::memcpy(&word, text + i, 4);
      std::ostringstream oss;
      oss << ".long 0x" << std::hex << word;
      translated_asm += oss.str() + "\n";
    }

    uint32_t num_vgprs12 = 8;
    uint32_t num_sgprs12 = 16;
    {
      uint32_t rsrc1_src = 0;
      if (kern.desc_offset != kern.code_offset &&
          kern.desc_offset + 52 <= elf_info.text_size)
        std::memcpy(&rsrc1_src, text + kern.desc_offset + 48, 4);
      else {
        for (const auto& sec : elf_info.sections) {
          if (sec.name == ".rodata" && sec.size >= 64) {
            for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
              const uint8_t* desc = elf + sec.offset + off;
              uint64_t entry;
              std::memcpy(&entry, desc + 16, 8);
              if (entry > 0 && entry < 1000000) { std::memcpy(&rsrc1_src, desc + 48, 4); break; }
            }
            break;
          }
        }
      }
      if (rsrc1_src) {
        num_vgprs12 = ((rsrc1_src & 0x3Fu) + 1u) * 12u;
        num_sgprs12 = (((rsrc1_src >> 6) & 0x3Fu) + 1u) * 16u;
        std::cerr << "hotswap: transpile: GFX12 RSRC1=0x" << std::hex
                  << rsrc1_src << std::dec << " → num_vgprs12=" << num_vgprs12
                  << " num_sgprs12=" << num_sgprs12 << "\n";
      }
    }
    if (num_vgprs12 < 8u) num_vgprs12 = 8u;
    if (num_sgprs12 < 16u) num_sgprs12 = 16u;

    // Scan MSGPACK for .sgpr_count
    {
      const char* key = ".sgpr_count";
      size_t key_len = 11;
      for (const auto& sec : elf_info.sections) {
        if (sec.name == ".note" && sec.size > key_len + 2) {
          for (size_t i = 0; i + key_len + 1 < sec.size; i++) {
            if (std::memcmp(elf + sec.offset + i, key, key_len) == 0) {
              uint8_t val = elf[sec.offset + i + key_len];
              uint32_t sgpr_count = 0;
              if (val <= 0x7F) sgpr_count = val;
              else if (val == 0xCC) sgpr_count = elf[sec.offset + i + key_len + 1];
              else if (val == 0xCD) {
                uint16_t v16;
                std::memcpy(&v16, elf + sec.offset + i + key_len + 1, 2);
                sgpr_count = (v16 >> 8) | ((v16 & 0xFF) << 8);
              }
              if (sgpr_count > num_sgprs12) num_sgprs12 = sgpr_count;
              break;
            }
          }
          break;
        }
      }
    }

    uint32_t save_vgpr_x = num_vgprs12;
    uint32_t save_vgpr_y = num_vgprs12 + 1u;
    uint32_t cmpx_temp_sgpr = num_sgprs12;
    const std::string sv_x = "v" + std::to_string(save_vgpr_x);
    const std::string sv_y = "v" + std::to_string(save_vgpr_y);

    uint64_t code_end = elf_info.text_size;
    if (ki + 1 < kernels.size())
      code_end = kernels[ki + 1].desc_offset;

    struct SourceInstr {
      std::string text;
      uint64_t pc_offset;
      uint32_t size;
      llvm::MCInst inst;
      bool valid_inst;
    };
    std::vector<SourceInstr> source_instrs;
    std::vector<std::string> source_lines;
    uint64_t pos = kern.code_offset;
    while (pos < code_end) {
      llvm::MCInst inst;
      uint64_t inst_size = 0;
      llvm::ArrayRef<uint8_t> bytes(text + pos, code_end - pos);
      auto status = src_state.disasm->getInstruction(inst, inst_size, bytes, pos, llvm::nulls());
      if (status == llvm::MCDisassembler::Fail) {
        if (pos + 4 <= code_end) {
          uint32_t word;
          std::memcpy(&word, text + pos, 4);
          std::ostringstream oss;
          oss << ".long 0x" << std::hex << word;
          source_instrs.push_back({oss.str(), pos, 4, llvm::MCInst(), false});
          source_lines.push_back(oss.str());
        }
        pos += 4;
        ++stats.total_instructions;
        continue;
      }
      std::string asm_text;
      if (src_state.printer) {
        llvm::raw_string_ostream rso(asm_text);
        src_state.printer->printInst(&inst, 0, "", *src_state.STI, rso);
        rso.flush();
      }
      size_t start = asm_text.find_first_not_of(" \t");
      if (start != std::string::npos && start > 0)
        asm_text = asm_text.substr(start);
      if (!asm_text.empty()) {
        source_instrs.push_back({asm_text, pos, static_cast<uint32_t>(inst_size), inst, true});
        source_lines.push_back(asm_text);
      }
      pos += inst_size;
      ++stats.total_instructions;
    }

    std::cerr << "hotswap: transpile: kernel " << ki << ": disassembled "
              << source_lines.size() << " instructions\n";

    // Branch label resolution
    std::map<uint64_t, std::string> branch_labels;
    int label_counter = 0;
    for (size_t i = 0; i < source_instrs.size(); ++i) {
      auto& info = source_instrs[i];
      std::string m = TranspileExtractMnemonic(info.text);
      bool is_branch = (m.find("s_branch") == 0 || m.find("s_cbranch_") == 0);
      if (!is_branch) continue;
      std::string ops = info.text.substr(info.text.find(m) + m.size());
      size_t s = ops.find_first_not_of(" \t");
      if (s == std::string::npos) continue;
      std::string offset_str = ops.substr(s);
      if (offset_str.find(".L_") == 0) continue;
      try {
        int64_t raw = std::stoll(offset_str, nullptr, 0);
        int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
        uint64_t target_pc = info.pc_offset + 4 + simm16 * 4;
        uint64_t snapped_pc = target_pc;
        bool found = false;
        for (const auto& si : source_instrs) {
          if (si.pc_offset >= target_pc) { snapped_pc = si.pc_offset; found = true; break; }
        }
        if (!found && !source_instrs.empty())
          snapped_pc = source_instrs.back().pc_offset;
        if (branch_labels.find(snapped_pc) == branch_labels.end())
          branch_labels[snapped_pc] = ".L_br" + std::to_string(label_counter++);
      } catch (...) {}
    }

    // TTMP taint analysis
    std::vector<SourceInstrForTaint> taint_input;
    taint_input.reserve(source_instrs.size());
    for (const auto& si : source_instrs)
      taint_input.push_back({si.text, si.inst, si.valid_inst});
    auto taint_results = AnalyzeTTMPTaint(taint_input, *src_state.MCII, *src_state.MRI);

    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace) {
        if (tr.replace_src == "v5") tr.replace_src = sv_x;
        else if (tr.replace_src == "v4") tr.replace_src = sv_y;
      }
    }

    translated_asm += "v_mov_b32_e32 " + sv_x + ", s2 ; save workgroup_id_x\n";
    translated_asm += "v_mov_b32_e32 " + sv_y + ", s3 ; save workgroup_id_y\n";

    std::vector<std::pair<std::string, std::string>> replace_regs;
    for (auto& tr : taint_results) {
      if (tr.action == TaintAction::Replace && !tr.replace_dst.empty())
        replace_regs.emplace_back(tr.replace_dst, tr.replace_src);
    }

    int early_exit_after = 0;
    if (const char* ee = std::getenv("HSA_HOTSWAP_EARLY_EXIT"))
      early_exit_after = std::atoi(ee);
    int emitted_count = 0;
    bool early_exit_done = false;

    for (size_t ii = 0; ii < source_lines.size(); ++ii) {
      const auto& line = source_lines[ii];

      if (ii < source_instrs.size()) {
        auto lbl = branch_labels.find(source_instrs[ii].pc_offset);
        if (lbl != branch_labels.end())
          translated_asm += lbl->second + ":\n";
      }

      if (ii < taint_results.size()) {
        if (taint_results[ii].action == TaintAction::Skip) continue;
        if (taint_results[ii].action == TaintAction::Replace) {
          auto& tr = taint_results[ii];
          translated_asm += "v_readfirstlane_b32 " + tr.replace_dst + ", " + tr.replace_src + "\n";
          continue;
        }
      }

      auto translated_lines = TranslateInstruction(line, src_cpu, tgt_cpu,
                                                    save_vgpr_y + 1, cmpx_temp_sgpr, false);

      if (ii < source_instrs.size() && !branch_labels.empty()) {
        for (auto& t : translated_lines) {
          std::string tm = TranspileExtractMnemonic(t);
          if (tm.find("s_branch") == 0 || tm.find("s_cbranch_") == 0) {
            size_t op_pos = t.find(tm) + tm.size();
            std::string ops = t.substr(op_pos);
            size_t s = ops.find_first_not_of(" \t");
            if (s != std::string::npos) {
              std::string off_str = ops.substr(s);
              if (off_str.find(".L_") != 0) {
                try {
                  int64_t raw = std::stoll(off_str, nullptr, 0);
                  int64_t simm16 = static_cast<int16_t>(raw & 0xFFFF);
                  uint64_t target = source_instrs[ii].pc_offset + 4 + simm16 * 4;
                  uint64_t snapped = target;
                  for (const auto& si : source_instrs) {
                    if (si.pc_offset >= target) { snapped = si.pc_offset; break; }
                  }
                  auto lbl = branch_labels.find(snapped);
                  if (lbl != branch_labels.end()) t = tm + " " + lbl->second;
                } catch (...) {}
              }
            }
          }
        }
      }

      bool translated_had_saveexec = false;
      for (const auto& t : translated_lines) {
        if (t.empty()) continue;
        if (t.find("saveexec") != std::string::npos) translated_had_saveexec = true;
        std::string m = TranspileExtractMnemonic(line);
        if (t.find("UNSUPPORTED") != std::string::npos) ++stats.unsupported_skipped;
        else if (t != line) {
          std::string nm = TranspileExtractMnemonic(t);
          if (IsWaitInstruction(m)) ++stats.translated_waitcnt;
          else if (nm != m) ++stats.translated_renamed;
          else if (t.find("exec_hi") != std::string::npos) ++stats.translated_exec;
          else ++stats.translated_passthrough;
        } else {
          ++stats.translated_passthrough;
        }
        translated_asm += t + "\n";
      }
      ++emitted_count;

      if (early_exit_after > 0 && emitted_count >= early_exit_after && !early_exit_done
          && source_lines.size() > 400) {
        for (size_t jj = ii + 1; jj < source_instrs.size(); jj++) {
          auto lbl = branch_labels.find(source_instrs[jj].pc_offset);
          if (lbl != branch_labels.end()) translated_asm += lbl->second + ":\n";
        }
        translated_asm += ".L_exit:\ns_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)\n";
        translated_asm += "s_endpgm ; EARLY EXIT after " + std::to_string(emitted_count) + " instrs\n";
        early_exit_done = true;
        break;
      }

      if (ii < source_instrs.size()) {
        bool has_vcmpx = false;
        for (const auto& t : translated_lines)
          if (t.find("v_cmpx_") != std::string::npos) { has_vcmpx = true; break; }
        if (has_vcmpx) {
          if (source_lines.size() > 400) {
            translated_asm += "s_mov_b32 exec_hi, 0\n";
          } else {
            bool next_is_execz = false;
            for (size_t nxt = ii + 1; nxt < source_instrs.size(); nxt++) {
              std::string nm = TranspileExtractMnemonic(source_instrs[nxt].text);
              if (nm.find("s_delay") == 0 || nm.find("s_wait") == 0 || nm.find("s_nop") == 0 || nm.find("s_clause") == 0) continue;
              if (nm == "s_cbranch_execz") next_is_execz = true;
              break;
            }
            if (!next_is_execz) translated_asm += "s_mov_b32 exec_hi, 0\n";
          }
        }
      }

      if (translated_had_saveexec) {
        for (auto& r : replace_regs)
          translated_asm += "v_readfirstlane_b32 " + r.first + ", " + r.second + "\n";
      }
    }
  }

  std::cerr << "hotswap: transpile: translated "
            << stats.total_instructions << " instructions → "
            << stats.translated_passthrough << " passthrough, "
            << stats.translated_renamed << " renamed, "
            << stats.translated_waitcnt << " waitcnt, "
            << stats.translated_exec << " exec-widened, "
            << stats.unsupported_skipped << " unsupported\n";

  // Post-processing
  {
    auto replaceAll = [](std::string& s, const std::string& from, const std::string& to) {
      size_t pos = 0;
      while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
      }
    };
    // VCC branch fix
    {
      std::string tmp;
      std::istringstream vfix_iss(translated_asm);
      std::string vfix_line;
      while (std::getline(vfix_iss, vfix_line)) {
        if (vfix_line.find("s_cbranch_vccz") != std::string::npos ||
            vfix_line.find("s_cbranch_vccnz") != std::string::npos)
          tmp += "s_mov_b32 vcc_hi, 0\n";
        tmp += vfix_line + "\n";
      }
      translated_asm = tmp;
    }
    replaceAll(translated_asm, "v_add_nc_u32 ", "v_add_u32_e32 ");
    replaceAll(translated_asm, "v_sub_nc_u32 ", "v_sub_u32_e32 ");
    // Strip explicit VCC mask from v_cndmask_b32_e32
    {
      std::string tmp;
      std::istringstream vcc_iss(translated_asm);
      std::string vcc_line;
      while (std::getline(vcc_iss, vcc_line)) {
        if (vcc_line.find("v_cndmask_b32_e32") != std::string::npos) {
          size_t vcc_pos = vcc_line.rfind(", vcc_lo");
          if (vcc_pos == std::string::npos) vcc_pos = vcc_line.rfind(", vcc");
          if (vcc_pos != std::string::npos)
            vcc_line = vcc_line.substr(0, vcc_pos);
        }
        tmp += vcc_line + "\n";
      }
      translated_asm = tmp;
    }
    // Fix s_load from s[8:9]+0xc with saved kernarg ptr
    {
      if (translated_asm.find("s_load_dword s1, s[8:9], 0xc") != std::string::npos ||
          translated_asm.find("s_load_dword s0, s[8:9], 0xc") != std::string::npos) {
        std::string ka_pair = "s[30:31]";
        size_t ka_pos = translated_asm.find("; save kernarg ptr lo");
        if (ka_pos != std::string::npos) {
          size_t s_pos = translated_asm.rfind("s_mov_b32 s", ka_pos);
          if (s_pos != std::string::npos) {
            size_t n_start = s_pos + 11;
            size_t n_end = translated_asm.find(',', n_start);
            int lo = std::stoi(translated_asm.substr(n_start, n_end - n_start));
            ka_pair = "s[" + std::to_string(lo) + ":" + std::to_string(lo + 1) + "]";
          }
        } else {
          size_t insert_pos = translated_asm.find("; save workgroup_id_y\n");
          if (insert_pos != std::string::npos) {
            insert_pos = translated_asm.find('\n', insert_pos) + 1;
            translated_asm.insert(insert_pos,
              "s_mov_b32 s30, s0 ; save kernarg ptr lo\n"
              "s_mov_b32 s31, s1 ; save kernarg ptr hi\n");
          }
        }
        replaceAll(translated_asm, "s_load_dword s1, s[8:9], 0xc",
                   "s_load_dword s1, " + ka_pair + ", 0x3c");
        replaceAll(translated_asm, "s_load_dword s0, s[8:9], 0xc",
                   "s_load_dword s0, " + ka_pair + ", 0x3c");
      }
    }
  }

  if (std::getenv("HSA_HOTSWAP_DUMP")) {
    std::cerr << "hotswap: transpile: === TRANSLATED ASSEMBLY ===\n"
              << translated_asm
              << "hotswap: transpile: === END ASSEMBLY ===\n";
  }

  // Assemble translated text for target ISA
  LLVMState &tgt_state = InitLLVMCached(target_isa);
  if (!tgt_state.valid) {
    std::cerr << "hotswap: transpile: failed to init target ISA '" << target_isa << "'\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  llvm::Triple tgt_triple("amdgcn-amd-amdhsa");
  llvm::MCTargetOptions mc_opts;

  tgt_state.Ctx->reset();

  llvm::StringRef asm_ref(translated_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

#if LLVM_VERSION_MAJOR > 14
  llvm::MCCodeEmitter* ce = tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.Ctx);
#else
  llvm::MCCodeEmitter* ce = tgt_state.target->createMCCodeEmitter(*tgt_state.MCII, *tgt_state.MRI, *tgt_state.Ctx);
#endif
  llvm::MCAsmBackend* mab = tgt_state.target->createMCAsmBackend(*tgt_state.STI, *tgt_state.MRI, mc_opts);

  if (!ce || !mab) {
    std::cerr << "hotswap: transpile: failed to create code emitter/backend\n";
    return AMD_COMGR_STATUS_ERROR;
  }

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      tgt_state.target->createMCObjectStreamer(
          tgt_triple, *tgt_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *tgt_state.STI,
          mc_opts.MCRelaxAll, mc_opts.MCIncrementalLinkerCompatible, false));
#endif

  if (!streamer) {
    std::cerr << "hotswap: transpile: failed to create MC streamer\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *tgt_state.Ctx, *streamer, *tgt_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      tgt_state.target->createMCAsmParser(*tgt_state.STI, *parser, *tgt_state.MCII, mc_opts));
  if (!tap) {
    std::cerr << "hotswap: transpile: failed to create target asm parser\n";
    return AMD_COMGR_STATUS_ERROR;
  }
  parser->setTargetParser(*tap);

  bool asm_failed = parser->Run(true);
  tap.reset();
  parser.reset();
  streamer.reset();
  bos.reset();
  data_stream->flush();

  if (asm_failed)
    std::cerr << "hotswap: transpile: assembly failed for " << tgt_cpu << "\n";

  if (data.size() < 64) {
    std::cerr << "hotswap: transpile: assembled output too small (" << data.size() << " bytes)\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  // Extract .text from assembled ELF
  const uint8_t* asm_elf = reinterpret_cast<const uint8_t*>(data.data());
  ElfInfo asm_info;
  if (!ParseElfInfo(asm_elf, data.size(), asm_info)) {
    std::cerr << "hotswap: transpile: failed to parse assembled ELF\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  const uint8_t* new_text = asm_elf + asm_info.text_offset;
  uint64_t new_text_size = asm_info.text_size;

  std::cerr << "hotswap: transpile: assembled " << new_text_size
            << " bytes (original: " << elf_info.text_size << ")\n";

  // Replace .text in a NEW writable ELF buffer
  {
    size_t new_elf_size = size;
    uint8_t* new_elf = static_cast<uint8_t*>(std::malloc(new_elf_size));
    if (!new_elf) return AMD_COMGR_STATUS_ERROR;
    std::memcpy(new_elf, elf, size);

    if (new_text_size <= elf_info.text_size) {
      std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
      uint8_t nop_bytes[] = {0x00, 0x00, 0x80, 0xBF};
      for (uint64_t i = new_text_size; i + 4 <= elf_info.text_size; i += 4)
        std::memcpy(new_elf + elf_info.text_offset + i, nop_bytes, 4);
    } else {
      uint64_t available = elf_info.text_size;
      uint64_t next_section_start = new_elf_size;
      uint16_t e_shentsize, e_shnum;
      std::memcpy(&e_shentsize, new_elf + 58, 2);
      std::memcpy(&e_shnum, new_elf + 60, 2);
      uint64_t e_shoff;
      std::memcpy(&e_shoff, new_elf + 40, 8);
      for (uint16_t i = 0; i < e_shnum; ++i) {
        uint64_t sh_off = e_shoff + i * e_shentsize;
        if (sh_off + e_shentsize > new_elf_size) break;
        uint64_t sec_offset, sec_size;
        std::memcpy(&sec_offset, new_elf + sh_off + 24, 8);
        std::memcpy(&sec_size, new_elf + sh_off + 32, 8);
        if (sec_offset > elf_info.text_offset && sec_offset < next_section_start && sec_size > 0)
          next_section_start = sec_offset;
      }
      available = next_section_start - elf_info.text_offset;

      if (new_text_size <= available) {
        std::memcpy(new_elf + elf_info.text_offset, new_text, new_text_size);
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          if (static_cast<int>(i) == elf_info.text_idx) {
            std::memcpy(new_elf + sh_off + 32, &new_text_size, 8);
            break;
          }
        }
      } else {
        uint64_t text_end = elf_info.text_offset + elf_info.text_size;
        uint64_t delta = ((new_text_size - elf_info.text_size + 255u) / 256u) * 256u;
        uint64_t grown_size = new_elf_size + delta;
        uint8_t* grown = (uint8_t*)calloc(1, grown_size);
        std::memcpy(grown, new_elf, text_end);
        std::memcpy(grown + elf_info.text_offset, new_text, new_text_size);
        uint64_t new_sec_size = elf_info.text_size + delta;
        for (uint64_t p = new_text_size; p < new_sec_size; p += 4) {
          uint8_t nop[] = {0x00, 0x00, 0x80, 0xBF};
          std::memcpy(grown + elf_info.text_offset + p, nop, 4);
        }
        uint64_t tail = new_elf_size - text_end;
        if (tail > 0) std::memcpy(grown + text_end + delta, new_elf + text_end, tail);
        std::memcpy(&e_shoff, grown + 40, 8);
        e_shoff += delta;
        std::memcpy(grown + 40, &e_shoff, 8);
        for (uint16_t i = 0; i < e_shnum; ++i) {
          uint64_t sh_off = e_shoff + i * e_shentsize;
          uint64_t sec_offset;
          std::memcpy(&sec_offset, grown + sh_off + 24, 8);
          if (sec_offset > elf_info.text_offset) {
            sec_offset += delta;
            std::memcpy(grown + sh_off + 24, &sec_offset, 8);
          }
          if (static_cast<int>(i) == elf_info.text_idx)
            std::memcpy(grown + sh_off + 32, &new_sec_size, 8);
        }
        uint64_t e_phoff;
        uint16_t e_phentsize, e_phnum;
        std::memcpy(&e_phoff, grown + 32, 8);
        std::memcpy(&e_phentsize, grown + 54, 2);
        std::memcpy(&e_phnum, grown + 56, 2);
        for (uint16_t i = 0; i < e_phnum; ++i) {
          uint64_t ph_off = e_phoff + i * e_phentsize;
          if (ph_off + 56 > grown_size) break;
          uint64_t p_offset, p_filesz, p_memsz;
          std::memcpy(&p_offset, grown + ph_off + 8, 8);
          std::memcpy(&p_filesz, grown + ph_off + 32, 8);
          std::memcpy(&p_memsz, grown + ph_off + 40, 8);
          if (p_offset == elf_info.text_offset) {
            p_filesz += delta; p_memsz += delta;
            std::memcpy(grown + ph_off + 32, &p_filesz, 8);
            std::memcpy(grown + ph_off + 40, &p_memsz, 8);
          } else if (p_offset > elf_info.text_offset) {
            p_offset += delta;
            std::memcpy(grown + ph_off + 8, &p_offset, 8);
          }
        }
        free(new_elf);
        new_elf = grown;
        new_elf_size = grown_size;
      }
    }

    *out_data = new_elf;
    *out_size = new_elf_size;

    // Patch kernel descriptors for wave64
    ElfInfo updated_info;
    if (ParseElfInfo(new_elf, new_elf_size, updated_info)) {
      PatchKernelDescriptorsForWave64(new_elf, new_elf_size, updated_info);
      for (auto& sec : updated_info.sections) {
        if (sec.name == ".rodata" && sec.size >= 64) {
          for (uint64_t off = 0; off + 64 <= sec.size; off += 64) {
            uint8_t* desc = new_elf + sec.offset + off;
            uint64_t entry;
            std::memcpy(&entry, desc + 16, 8);
            if (entry == 0 || entry > 1000000) continue;
            uint32_t rsrc1;
            std::memcpy(&rsrc1, desc + 48, 4);
            rsrc1 &= 0x00FFFFFFu;
            rsrc1 |= (1u << 21) | (1u << 23);
            uint32_t vgpr_field12 = rsrc1 & 0x3Fu;
            uint32_t sgpr_field12_rd = (rsrc1 >> 6) & 0x3Fu;
            uint32_t num_vgprs = (vgpr_field12 + 1u) * 12u;
            if (num_vgprs < 8u) num_vgprs = 8u;
            num_vgprs += 4u;
            uint32_t gfx9_vgpr = (num_vgprs / 4u) - 1u;
            if (gfx9_vgpr > 62u) gfx9_vgpr = 62u;
            uint32_t rsrc1_vgpr_field = gfx9_vgpr;
            if (rsrc1_vgpr_field > 63u) rsrc1_vgpr_field = 63u;
            rsrc1 &= ~0xFFFu;
            rsrc1 |= (rsrc1_vgpr_field << 6u);
            {
              uint32_t num_sgprs_rd = (sgpr_field12_rd + 1u) * 16u + 8u;
              const char* sgpr_key = ".sgpr_count";
              for (size_t si = 0; si + 12 < new_elf_size; si++) {
                if (std::memcmp(new_elf + si, sgpr_key, 11) == 0) {
                  uint8_t val = new_elf[si + 11];
                  uint32_t sc = (val <= 0x7F) ? val : (val == 0xCC ? new_elf[si+12] : 0);
                  if (sc + 8 > num_sgprs_rd) num_sgprs_rd = sc + 8;
                  break;
                }
              }
              uint32_t gfx9_sgpr = (num_sgprs_rd / 8u) - 1u;
              if (gfx9_sgpr > 12u) gfx9_sgpr = 12u;
              rsrc1 |= gfx9_sgpr;
            }
            std::memcpy(desc + 48, &rsrc1, 4);
            uint32_t rsrc2;
            std::memcpy(&rsrc2, desc + 52, 4);
            rsrc2 |= (1u << 7) | (1u << 8) | (1u << 9);
            std::memcpy(desc + 52, &rsrc2, 4);
            uint16_t props;
            std::memcpy(&props, desc + 56, 2);
            props = static_cast<uint16_t>(
                (static_cast<uint32_t>(props) & ~(1u << 10)) & 0xFFFFu);
            std::memcpy(desc + 56, &props, 2);
            uint32_t rsrc3 = gfx9_vgpr;
            std::memcpy(desc + 44, &rsrc3, 4);
          }
        }
      }
    }

    // Patch ELF metadata
    PatchElfMetadata(new_elf, new_elf_size, tgt_cpu);
  }

  result->rules_matched = stats.translated_passthrough + stats.translated_renamed + stats.translated_waitcnt;

  std::cerr << "hotswap: transpile: complete (" << src_cpu << " → " << tgt_cpu << ")\n";

  if (auto* dump_path = std::getenv("HSA_HOTSWAP_DUMP_ELF")) {
    FILE* fp = fopen(dump_path, "wb");
    if (fp) {
      fwrite(*out_data, 1, *out_size, fp);
      fclose(fp);
      std::cerr << "hotswap: transpile: dumped patched ELF to " << dump_path << "\n";
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
