//===- translation_cache_test.cpp - translation_cache unit tests ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "hotswap/translation-cache.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TempDir {
  llvm::SmallString<128> Path;
  bool Valid = false;

  explicit TempDir(const char *Prefix) {
    std::error_code Ec = llvm::sys::fs::createUniqueDirectory(Prefix, Path);
    Valid = !Ec;
  }

  ~TempDir() {
    if (Valid)
      llvm::sys::fs::remove_directories(Path);
  }

  std::string file(const char *Name) const {
    llvm::SmallString<256> P(Path);
    llvm::sys::path::append(P, Name);
    return std::string(P);
  }
};

struct ScopedEnv {
  std::string Name;
  std::string OldValue;
  bool HadOldValue = false;

  ScopedEnv(const char *Name, const std::string &Value) : Name(Name) {
    if (const char *Old = std::getenv(Name)) {
      OldValue = Old;
      HadOldValue = true;
    }
    setenv(Name, Value.c_str(), 1);
  }

  ~ScopedEnv() {
    if (HadOldValue)
      setenv(Name.c_str(), OldValue.c_str(), 1);
    else
      unsetenv(Name.c_str());
  }
};

llvm::MemoryBufferRef bufRef(const std::vector<uint8_t> &V) {
  return llvm::MemoryBufferRef(
      llvm::StringRef(reinterpret_cast<const char *>(V.data()), V.size()), "");
}

std::vector<uint8_t> fakeAmdgpuElf() {
  std::vector<uint8_t> Data(128, 0);
  Data[0] = 0x7f;
  Data[1] = 'E';
  Data[2] = 'L';
  Data[3] = 'F';
  Data[4] = 2; // ELFCLASS64
  Data[5] = 1; // little-endian
  Data[6] = 1; // current ELF version
  const uint16_t Machine = 224; // EM_AMDGPU
  const uint32_t Flags = 0x49;
  std::memcpy(Data.data() + 18, &Machine, sizeof(Machine));
  std::memcpy(Data.data() + 48, &Flags, sizeof(Flags));
  return Data;
}

void writeTextFile(const std::string &Path, llvm::StringRef Text) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os << Text;
}

void writeBinaryFile(const std::string &Path,
                     const std::vector<uint8_t> &Bytes) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec, llvm::sys::fs::OF_None);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os.write(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
}

COMGR::hotswap::TranslationCacheRequest makeRequest(
    llvm::MemoryBufferRef Source, const std::string &RulesPath,
    const std::string &SourceGfx = "gfx1250",
    const std::string &TargetGfx = "gfx942") {
  COMGR::hotswap::TranslationCacheRequest Request;
  Request.SourceObject = Source;
  Request.SourceGfx = SourceGfx;
  Request.TargetGfx = TargetGfx;
  Request.SourceIsa = "amdgcn-amd-amdhsa--" + SourceGfx;
  Request.TargetIsa = "amdgcn-amd-amdhsa--" + TargetGfx;
  Request.CodeIsa = "amdgcn-amd-amdhsa--gfx942";
  Request.HotswapRulesPath = RulesPath;
  Request.CacheDirectory = llvm::sys::path::parent_path(RulesPath).str();
  Request.CacheDisabled = false;
  Request.OrigMach = 0x49;
  Request.EnableWritelaneRewrite = true;
  Request.EnableWaveNative = true;
  Request.StrictMode = true;
  return Request;
}

COMGR::hotswap::PipelineResult makeSuccessfulResult(
    std::vector<uint8_t> Hsaco = {0x7f, 'E', 'L', 'F', 1, 2, 3}) {
  COMGR::hotswap::PipelineResult Result;
  Result.Success = true;
  Result.Hsaco = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(reinterpret_cast<const char *>(Hsaco.data()),
                      Hsaco.size()),
      "");
  Result.LiftedCount = 7;
  Result.TotalCount = 7;
  return Result;
}

} // namespace

TEST(TranslationCache, FirstRunMissWriteSecondRunHit) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);

  auto First = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(First.Status, COMGR::hotswap::TranslationCacheStatus::Miss);

  auto Result = makeSuccessfulResult();
  auto Write = COMGR::hotswap::writeTranslationCache(Request, Result);
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess)
      << Write.Reason;

  auto Second = COMGR::hotswap::lookupTranslationCache(Request);
  ASSERT_EQ(Second.Status, COMGR::hotswap::TranslationCacheStatus::Hit)
      << Second.Reason;
  ASSERT_TRUE(Second.Result.Hsaco && Result.Hsaco);
  EXPECT_EQ(Second.Result.Hsaco->getBuffer(), Result.Hsaco->getBuffer());
  EXPECT_EQ(Second.Result.LiftedCount, Result.LiftedCount);
  EXPECT_EQ(Second.Result.TotalCount, Result.TotalCount);
}

TEST(TranslationCache, ChangedInputHashCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  ASSERT_EQ(COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult()).Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  Source[80] ^= 0x1;
  auto Changed = makeRequest(bufRef(Source), Rules);
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Changed);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, ChangedIsaCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  ASSERT_EQ(COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult()).Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  auto ChangedSourceIsa = makeRequest(bufRef(Source), Rules, "gfx1200", "gfx942");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedSourceIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);

  auto ChangedTargetIsa = makeRequest(bufRef(Source), Rules, "gfx1250", "gfx950");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedTargetIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, OldHotswapCacheDirDoesNotEnableCache) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv OldCacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", "");
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  Request.CacheDirectory = "";

  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Disabled);
}

TEST(TranslationCache, CorruptMetadataIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  auto Write = COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeTextFile(Write.MetadataPath, "not-json\n");
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("parse"), std::string::npos);
}

TEST(TranslationCache, CorruptObjectIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  auto Write = COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeBinaryFile(Write.ObjectPath, {1, 2, 3, 4});
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("cached_object_sha256"), std::string::npos);
}

TEST(TranslationCache, ReadonlyMissDoesNotWrite) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(bufRef(Source), Rules);
  Request.CacheReadonly = true;

  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);

  auto Write = COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  EXPECT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::Disabled);

  auto Second = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Second.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, BypassedStatusHasStableString) {
  EXPECT_STREQ(COMGR::hotswap::translationCacheStatusString(
                   COMGR::hotswap::TranslationCacheStatus::Bypassed),
               "bypassed");
}

TEST(TranslationCache, SkipKernelListMatchesExactKernelName) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS",
                 "other_kernel, target_kernel ,third_kernel");

  std::vector<std::string> Kernels = {"first_kernel", "target_kernel"};
  EXPECT_EQ(COMGR::hotswap::skippedKernelForTranslationCache(
                Kernels, "other_kernel, target_kernel ,third_kernel"),
            "target_kernel");
}

TEST(TranslationCache, SkipKernelListDoesNotUseSubstringMatching) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS", "target");

  std::vector<std::string> Kernels = {"target_kernel"};
  EXPECT_TRUE(
      COMGR::hotswap::skippedKernelForTranslationCache(Kernels, "target").empty());
}
