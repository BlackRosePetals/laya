// laya::tokenizer against the `tokenizers` crate (through the Python-recorded goldens) and against
// hand-built tokenizers. Ported from laya-go/tokenizer's tests.
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>

#include "golden.hpp"
#include "laya/pyvalue.hpp"
#include "laya/sha256.hpp"
#include "laya/tokenizer.hpp"
#include "laya/unicodedata.hpp"

using laya::tokenizer::Form;
using laya::tokenizer::Tokenizer;
using laya_test::At;
using laya_test::Value;
namespace tk = laya::tokenizer;
namespace ud = laya::unicodedata;
namespace pv = laya::pyvalue;

namespace {

const char* const kShipped[] = {"english", "multilingual"};

std::vector<int32_t> Ints(const Value& v) {
  std::vector<int32_t> out;
  for (const Value& x : v.items()) out.push_back(int32_t(x.as_int()));
  return out;
}
std::string Join(const std::vector<std::string>& v) {
  std::string out;
  for (const auto& s : v) out += s + "\x01";
  return out;
}
std::string Ids(const std::vector<int32_t>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size() && i < 24; i++) out += std::to_string(v[i]) + " ";
  return out + (v.size() > 24 ? "... " : "") + "(" + std::to_string(v.size()) + ")]";
}
std::string Hex(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    char b[4];
    std::snprintf(b, sizeof b, "%02x", c);
    out += b;
  }
  return out;
}

// A shipped checkpoint's tokenizer, loaded once. Without LAYA_CHECKPOINTS the test is skipped
// locally and FAILS under LAYA_REQUIRE_GOLDENS=1 (CI), so a missing checkpoint cannot pass as green.
const Tokenizer* Checkpoint(const std::string& name) {
  static std::mutex mu;
  static std::map<std::string, Tokenizer> loaded;
  std::lock_guard<std::mutex> lock(mu);
  auto it = loaded.find(name);
  if (it != loaded.end()) return &it->second;
  const char* root = std::getenv("LAYA_CHECKPOINTS");
  if (root == nullptr || *root == 0) return nullptr;
  auto t = Tokenizer::LoadDir(std::string(root) + "/" + name);
  if (!t.ok()) {
    ADD_FAILURE() << "load the " << name << " tokenizer: " << t.error();
    return nullptr;
  }
  return &loaded.emplace(name, std::move(t).value()).first->second;
}

#define LAYA_CHECKPOINT(var, name)                                                         \
  const Tokenizer* var = Checkpoint(name);                                                 \
  if (var == nullptr) {                                                                    \
    const char* req = std::getenv("LAYA_REQUIRE_GOLDENS");                                 \
    if (req != nullptr && std::string(req) == "1" && !HasFailure()) {                      \
      ADD_FAILURE() << "LAYA_CHECKPOINTS is unset but LAYA_REQUIRE_GOLDENS=1";             \
    } else if (!HasFailure()) {                                                            \
      GTEST_SKIP() << "LAYA_CHECKPOINTS unset (local run)";                                \
    }                                                                                      \
    return;                                                                                \
  }

Tokenizer MustParse(const std::string& json, const char* config = nullptr) {
  auto t = Tokenizer::Parse(json, config == nullptr ? std::nullopt : std::optional<std::string_view>(config));
  EXPECT_TRUE(t.ok()) << (t.ok() ? "" : t.error());
  if (!t.ok()) std::abort();
  return std::move(t).value();
}

// A tiny byte-level tokenizer built in the test, so the loader is checked without a checkpoint.
std::string ByteLevelSpec(const std::string& extra) {
  std::string vocab;
  for (int b = 0, lifted = 0; b < 256; b++) {
    char32_t r;
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) r = char32_t(b);
    else r = char32_t(256 + lifted++);
    std::string key;
    ud::AppendRune(key, r);
    if (r == '"' || r == '\\') key = "\\" + key;
    vocab += "\"" + key + "\":" + std::to_string(b) + ",";
  }
  vocab += "\"ab\":256";
  return "{\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab + "},\"merges\":[\"a b\"]},\"normalizer\":null," +
         "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true}" + extra + "}";
}
std::string Replace(std::string s, const std::string& from, const std::string& to) {
  size_t at = s.find(from);
  if (at != std::string::npos) s.replace(at, from.size(), to);
  return s;
}

std::string Digest(const std::function<void(const std::function<void(const std::string&)>&)>& rows) {
  laya::Sha256 h;
  rows([&](const std::string& s) { h.Update(s); });
  return laya::ToHex(h.Finish());
}
std::string Mapped(char32_t cp, const std::string& text) {
  std::string out = std::to_string(uint32_t(cp)) + ":";
  bool first = true;
  std::string_view rest(text);
  while (!rest.empty()) {
    auto d = ud::DecodeRune(rest);
    out += (first ? "" : ",") + std::to_string(uint32_t(d.rune));
    first = false;
    rest.remove_prefix(d.width);
  }
  return out + ";";
}
std::string One(char32_t cp) {
  std::string s;
  ud::AppendRune(s, cp);
  return s;
}
const char* PyBool(bool b) { return b ? "True" : "False"; }

}  // namespace

// ---- normalization, lowercase and the single_word flank over every code point ----------------------
class FormSweep : public ::testing::TestWithParam<std::pair<const char*, Form>> {};
TEST_P(FormSweep, MatchesTheCrateOverAllUnicode) {
  LAYA_GOLDEN_LIST(sweep, "go_extra.json", "normalize", "sweep");
  Form form = GetParam().second;
  std::string got = Digest([&](const auto& yield) {
    for (char32_t r = 0; r < 0x110000; r++) {
      if (r >= 0xD800 && r <= 0xDFFF) continue;
      yield(Mapped(r, tk::Normalize(One(r), form)));
    }
  });
  EXPECT_EQ(got, sweep->Find(GetParam().first)->as_string()) << "some single code point differs from the crate";
}
INSTANTIATE_TEST_SUITE_P(Golden, FormSweep,
                         ::testing::Values(std::make_pair("NFC", Form::NFC), std::make_pair("NFD", Form::NFD),
                                           std::make_pair("NFKC", Form::NFKC), std::make_pair("NFKD", Form::NFKD)));

TEST(TokenizerNormalize, RandomStringsMatchTheCrate) {
  LAYA_GOLDEN_LIST(cases, "go_extra.json", "normalize", "cases");
  ASSERT_GE(cases->size(), 1000u);
  for (size_t i = 0; i < cases->size(); i++) {
    const Value& c = cases->value(i);
    const std::string& text = c.Find("text")->as_string();
    const std::pair<const char*, Form> forms[] = {{"nfc", Form::NFC}, {"nfd", Form::NFD}, {"nfkc", Form::NFKC}, {"nfkd", Form::NFKD}};
    for (const auto& f : forms) {
      ASSERT_EQ(tk::Normalize(text, f.second), c.Find(f.first)->as_string()) << "case " << i << " " << f.first << " of " << Hex(text);
    }
  }
}

TEST(TokenizerNormalize, LowercaseMatchesTheCrateOverAllUnicode) {
  LAYA_GOLDEN_LIST(want, "go_extra.json", "lowercase", "sweep");
  std::string got = Digest([&](const auto& yield) {
    for (char32_t r = 0; r < 0x110000; r++) {
      if (r >= 0xD800 && r <= 0xDFFF) continue;
      yield(Mapped(r, tk::Lowercase(One(r))));
    }
  });
  EXPECT_EQ(got, want->as_string());
  EXPECT_EQ(tk::Lowercase("\xCE\xA3"), "\xCF\x83") << "no final-sigma rule: the crate lowercases per character";
}

// A single_word added token is refused beside a word character. Which characters those are is the
// crate's rule, not an alphanumeric test: it counts combining marks and joiners and not other numbers.
TEST(TokenizerAdded, SingleWordFlankMatchesTheCrate) {
  LAYA_GOLDEN_LIST(sw, "go_extra.json", "single_word");
  Tokenizer tok = MustParse(
      R"({"model":{"type":"BPE","vocab":{"a":0,"<unk>":2},"merges":[],"unk_token":"<unk>"},
        "added_tokens":[{"content":"TOK","id":1,"single_word":true,"special":true}]})");
  auto standalone = [&](const std::string& s) {
    for (int32_t id : tok.Encode(s)) {
      if (id == 1) return std::string("1");
    }
    return std::string("0");
  };
  for (const char* side : {"left_standalone", "right_standalone"}) {
    bool left = std::string(side) == "left_standalone";
    std::string got = Digest([&](const auto& yield) {
      for (char32_t r = 0; r < 0x110000; r++) {
        if (r >= 0xD800 && r <= 0xDFFF) {
          yield("0");
          continue;
        }
        yield(standalone(left ? One(r) + "TOK" : "TOK" + One(r)));
      }
    });
    EXPECT_EQ(got, sw->Find(side)->as_string()) << side;
  }
}

// ---- pre-tokenizers and normalizer specs ---------------------------------------------------------------
TEST(TokenizerPre, PreTokenizersMatchTheCrate) {
  LAYA_GOLDEN_LIST(pre, "go_extra.json", "pretokenize");
  const Value& texts = *pre->Find("texts");
  size_t checked = 0;
  auto check = [&](const std::string& name, const std::string& spec, const Value& want) {
    ASSERT_EQ(want.size(), texts.size()) << name;
    for (size_t i = 0; i < texts.size(); i++) {
      auto got = tk::RunPreTokenizerSpec(spec, texts.value(i).as_string());
      ASSERT_TRUE(got.ok()) << got.error();
      std::vector<std::string> w;
      for (const Value& x : want.value(i).items()) w.push_back(x.as_string());
      ASSERT_EQ(Join(got.value()), Join(w)) << name << " of " << Hex(texts.value(i).as_string());
      checked++;
    }
  };
  for (bool prefix : {false, true}) {
    for (bool regex : {true, false}) {
      std::string key = std::string("add_prefix_space=") + PyBool(prefix) + ",use_regex=" + PyBool(regex);
      check("ByteLevel/" + key,
            std::string("{\"type\":\"ByteLevel\",\"add_prefix_space\":") + (prefix ? "true" : "false") + ",\"use_regex\":" + (regex ? "true" : "false") + "}",
            *At(pre, {"byte_level", key.c_str()}));
    }
  }
  for (const char* scheme : {"always", "first", "never"}) {
    for (bool split : {true, false}) {
      std::string key = std::string(scheme) + ",split=" + PyBool(split);
      check("Metaspace/" + key,
            std::string("{\"type\":\"Metaspace\",\"replacement\":\"\xE2\x96\x81\",\"prepend_scheme\":\"") + scheme + "\",\"split\":" + (split ? "true" : "false") + "}",
            *At(pre, {"metaspace", key.c_str()}));
    }
  }
  EXPECT_GE(checked, 10u * 1000u);
}

TEST(TokenizerNormalize, NormalizerSpecsMatchTheCrate) {
  LAYA_GOLDEN_LIST(fam, "go_extra.json", "normalizers");
  const Value& inputs = *fam->Find("inputs");
  const Value& specs = *fam->Find("specs");
  ASSERT_GE(specs.size(), 12u);
  for (size_t s = 0; s < specs.size(); s++) {
    auto spec = pv::Dumps(*specs.value(s).Find("spec"));
    ASSERT_TRUE(spec.ok());
    const Value& outs = *specs.value(s).Find("outputs");
    for (size_t j = 0; j < inputs.size(); j++) {
      auto got = tk::RunNormalizerSpec(spec.value(), inputs.value(j).as_string());
      ASSERT_TRUE(got.ok()) << spec.value() << ": " << got.error();
      ASSERT_EQ(got.value(), outs.value(j).as_string()) << spec.value() << " on " << Hex(inputs.value(j).as_string());
    }
  }
}

// ---- the shipped tokenizers against Python ------------------------------------------------------------
class Shipped : public ::testing::TestWithParam<const char*> {};
INSTANTIATE_TEST_SUITE_P(Checkpoint, Shipped, ::testing::ValuesIn(kShipped));

TEST_P(Shipped, CorpusIdsIdentityAndSpecialsMatchPython) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(side, "tokenizer_ids.json", name.c_str());
  LAYA_CHECKPOINT(tok, name);
  const Value* corpus_text = At(laya_test::Load("tokenizer_ids.json", true), {"corpus_text"});
  ASSERT_NE(corpus_text, nullptr);
  const Value& id = *side->Find("identity");
  size_t want_vocab = size_t(id.Find("vocab_size")->as_int());
  if (const Value* extra = id.Find("added_tokens_outside_vocab"); extra != nullptr && !extra->is_null()) want_vocab += size_t(extra->as_int());
  EXPECT_EQ(tok->VocabSize(), want_vocab);
  const Value& sp = *side->Find("special_ids");
  EXPECT_EQ(tok->Cls(), std::optional<int32_t>(int32_t(sp.Find("cls")->as_int())));
  EXPECT_EQ(tok->Sep(), std::optional<int32_t>(int32_t(sp.Find("sep")->as_int())));
  EXPECT_EQ(tok->Mask(), std::optional<int32_t>(int32_t(sp.Find("mask")->as_int())));
  EXPECT_EQ(tok->Pad(), std::optional<int32_t>(int32_t(sp.Find("pad")->as_int())));
  EXPECT_EQ(tok->MaskToken(), sp.Find("mask_token")->as_string());
  const Value& corpus = *side->Find("corpus");
  ASSERT_GE(corpus.size(), 35u);
  for (size_t i = 0; i < corpus.size(); i++) {
    const std::string& text = corpus_text->Find(corpus.key(i))->as_string();
    auto got = tok->Encode(text);
    EXPECT_EQ(got, Ints(corpus.value(i))) << corpus.key(i) << " " << Ids(got);
  }
  const Value& trunc = *side->Find("option_truncation");
  std::string text = " " + trunc.Find("text")->as_string();
  EXPECT_EQ(tok->Encode(text), Ints(*trunc.Find("uncapped")));
  EXPECT_EQ(tok->Encode(text, 48), Ints(*trunc.Find("capped_48")));
}

TEST_P(Shipped, DecodeMatchesPythonWholeAndSliced) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(entry, "decode_text.json", "by", name.c_str());
  LAYA_CHECKPOINT(tok, name);
  const Value& cases = *entry->Find("cases");
  ASSERT_GE(cases.size(), 23u);
  size_t slices = 0;
  for (const Value& c : cases.items()) {
    const std::string& text = c.Find("text")->as_string();
    auto ids = Ints(*c.Find("ids"));
    EXPECT_EQ(tok->Encode(text), ids) << Hex(text);
    EXPECT_EQ(tok->Decode(ids, true), c.Find("decoded")->as_string());
    for (const Value& s : c.Find("slices")->items()) {
      slices++;
      size_t start = size_t(s.Find("start")->as_int());
      size_t end = std::min(start + size_t(s.Find("size")->as_int()), ids.size());
      std::vector<int32_t> part(ids.begin() + long(start), ids.begin() + long(end));
      EXPECT_EQ(tok->Decode(part, true), s.Find("decoded")->as_string()) << "slice " << start;
    }
    if (c.Find("slices")->size() == 0) EXPECT_TRUE(text.empty() && ids.empty());
    const Value& w = *c.Find("with_specials");
    auto with = Ints(*w.Find("ids"));
    EXPECT_EQ(tok->Decode(with, true), w.Find("skipping_specials")->as_string());
    EXPECT_EQ(tok->Decode(with, false), w.Find("kept")->as_string());
  }
  EXPECT_GE(slices, 150u) << "the slice cases are the point of this family";
}

TEST_P(Shipped, EncodeFuzzMatchesTheCrate) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(side, "go_extra.json", "encode_fuzz", name.c_str());
  LAYA_CHECKPOINT(tok, name);
  const Value& texts = *side->Find("texts");
  const Value& ids = *side->Find("ids");
  ASSERT_GE(texts.size(), 500u);
  int bad = 0;
  for (size_t i = 0; i < texts.size() && bad < 5; i++) {
    auto got = tok->Encode(texts.value(i).as_string());
    if (got != Ints(ids.value(i))) {
      ADD_FAILURE() << "Encode(" << Hex(texts.value(i).as_string()) << ") = " << Ids(got) << ", want " << Ids(Ints(ids.value(i)));
      bad++;
    }
  }
}

TEST_P(Shipped, DecodeFuzzMatchesTheCrate) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(cases, "go_extra.json", "decode_fuzz", name.c_str(), "cases");
  LAYA_CHECKPOINT(tok, name);
  int bad = 0;
  for (const Value& c : cases->items()) {
    auto ids = Ints(*c.Find("ids"));
    for (bool skip : {true, false}) {
      if (tok->Decode(ids, skip) != c.Find(skip ? "skipping" : "keeping")->as_string() && bad++ < 5) {
        ADD_FAILURE() << "Decode(" << Ids(ids) << ", skip=" << skip << ")";
      }
    }
  }
}

// A Python str is always valid, so invalid bytes reach the tokenizer as bytes.decode('utf-8','replace').
TEST_P(Shipped, InvalidUtf8EncodesLikePython) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(fam, "go_extra.json", "invalid_utf8");
  LAYA_CHECKPOINT(tok, name);
  const Value& hexes = *fam->Find("hex");
  const Value& want = *At(fam, {"ids", name.c_str()});
  ASSERT_GE(want.size(), 600u);
  for (size_t i = 0; i < want.size(); i++) {
    const std::string& h = hexes.value(i).as_string();
    std::string raw;
    for (size_t k = 0; k + 1 < h.size(); k += 2) raw.push_back(char(std::stoi(h.substr(k, 2), nullptr, 16)));
    ASSERT_EQ(tok->Encode(raw), Ints(want.value(i))) << h;
  }
}

// A loaded Tokenizer is read-only, so any number of threads may share it.
TEST_P(Shipped, ConcurrentUse) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(side, "go_extra.json", "encode_fuzz", name.c_str());
  LAYA_CHECKPOINT(tok, name);
  const Value& texts = *side->Find("texts");
  const Value& ids = *side->Find("ids");
  std::vector<std::string> errors(8);
  std::vector<std::thread> threads;
  for (int g = 0; g < 8; g++) {
    threads.emplace_back([&, g] {
      for (size_t i = size_t(g); i < texts.size(); i += size_t(1 + g % 3)) {
        auto got = tok->Encode(texts.value(i).as_string());
        if (got != Ints(ids.value(i))) {
          errors[size_t(g)] = "thread " + std::to_string(g) + " text " + std::to_string(i);
          return;
        }
        (void)tok->Decode(got, true);
      }
    });
  }
  for (auto& t : threads) t.join();
  for (const auto& e : errors) EXPECT_EQ(e, "");
}

TEST(TokenizerDecode, DecoderChainsAndEdges) {
  LAYA_GOLDEN_LIST(by, "decode_text.json", "by");
  EXPECT_EQ(At(by, {"english", "decoder", "type"})->as_string(), "ByteLevel");
  std::string types;
  for (const Value& s : At(by, {"multilingual", "decoder", "decoders"})->items()) types += s.Find("type")->as_string() + ",";
  EXPECT_EQ(types, "Replace,ByteFallback,Fuse,");
}

TEST(TokenizerDecode, SyntheticDecoderShapesAndMixedToken) {
  LAYA_GOLDEN_LIST(shapes, "decode_text.json", "synthetic_decoders");
  ASSERT_GE(shapes->size(), 5u);
  for (size_t i = 0; i < shapes->size(); i++) {
    const Value& shape = shapes->value(i);
    auto spec = pv::Dumps(*shape.Find("spec"));
    ASSERT_TRUE(spec.ok());
    Tokenizer tok = MustParse(spec.value());
    ASSERT_GT(shape.Find("cases")->size(), 0u);
    for (const Value& c : shape.Find("cases")->items()) {
      EXPECT_EQ(tok.Decode(Ints(*c.Find("ids")), false), c.Find("decoded")->as_string()) << shapes->key(i);
    }
    if (const Value* panics = shape.Find("crate_panics")) {  // the crate panics on these; this port returns empty text
      for (const Value& p : panics->items()) EXPECT_EQ(tok.Decode(Ints(p), false), "") << shapes->key(i);
    }
  }
  const Value* mixed = At(laya_test::Load("decode_text.json", true), {"mixed_token"});
  ASSERT_NE(mixed, nullptr);
  EXPECT_EQ(mixed->Find("scope")->as_string(), "token");
  auto spec = pv::Dumps(*mixed->Find("spec"));
  ASSERT_TRUE(spec.ok());
  Tokenizer tok = MustParse(spec.value());
  for (const Value& c : mixed->Find("cases")->items()) {
    EXPECT_EQ(tok.Decode(Ints(*c.Find("ids")), false), c.Find("decoded")->as_string()) << c.Find("why")->as_string();
  }
}

TEST(TokenizerDecode, EdgesOnTheEnglishAndMultilingualCheckpoints) {
  LAYA_CHECKPOINT(en, "english");
  LAYA_CHECKPOINT(ml, "multilingual");
  EXPECT_EQ(en->Decode({}, true), "");
  EXPECT_EQ(en->Decode({1 << 30}, true), "");
  int32_t a = en->Encode("a")[0];
  EXPECT_EQ(en->Decode({1 << 30, a, -1}, true), "a");
  EXPECT_EQ(en->Decode({50275}, true), "   ") << "the literal-whitespace added token survives decoding";
  for (const char* s : {" ", "   "}) EXPECT_EQ(en->Decode(en->Encode(s), true), s);
  EXPECT_EQ(ml->Decode(ml->Encode("Hello world"), true), " Hello world");
  auto ids = ml->Encode("\xF3\xB0\x80\x80");  // U+F0000, four bytes of byte fallback
  ASSERT_GE(ids.size(), 3u);
  EXPECT_EQ(ml->Decode({ids[0], ids[1], ids[2]}, true), " \xEF\xBF\xBD\xEF\xBF\xBD");
  EXPECT_EQ(ml->Decode(ids, true), " \xF3\xB0\x80\x80");
  auto full = en->Encode("hello world, this is a longer sentence");
  EXPECT_TRUE(en->Encode("hello world", 0).empty());
  for (size_t n = 1; n <= full.size(); n++) {
    EXPECT_EQ(en->Encode("hello world, this is a longer sentence", int64_t(n)), std::vector<int32_t>(full.begin(), full.begin() + long(n)));
  }
  EXPECT_EQ(en->Encode("hello", 100), en->Encode("hello"));
}

// ---- loader behaviour on hand-built tokenizers ------------------------------------------------------
TEST(TokenizerLoader, SmallTokenizerRoundTrips) {
  Tokenizer tok = MustParse(ByteLevelSpec(",\"decoder\":{\"type\":\"ByteLevel\"}"));
  auto ids = tok.Encode("ab ab");
  EXPECT_EQ(tok.Decode(ids, true), "ab ab");
  EXPECT_EQ(ids[0], 256);
  EXPECT_EQ(tok.VocabSize(), 257u);
  EXPECT_EQ(tok.HighestId(), 256);
  EXPECT_EQ(tok.IdToToken(256), std::optional<std::string>("ab"));
  EXPECT_FALSE(tok.IdToToken(-1).has_value());
  EXPECT_EQ(tok.TokenToId("ab"), std::optional<int32_t>(256));
}

TEST(TokenizerLoader, Refuses) {
  std::string good = ByteLevelSpec("");
  struct Case {
    const char* name;
    std::string spec;
    bool unsupported;
  } cases[] = {
      {"a non-BPE model", Replace(good, "\"BPE\"", "\"WordPiece\""), true},
      {"an unknown normalizer", Replace(good, "\"normalizer\":null", "\"normalizer\":{\"type\":\"BertNormalizer\"}"), true},
      {"an unknown pre-tokenizer", Replace(good, "\"pre_tokenizer\":{\"type\":\"ByteLevel\"", "\"pre_tokenizer\":{\"type\":\"Whitespace\""), true},
      {"an unknown decoder", ByteLevelSpec(",\"decoder\":{\"type\":\"WordPiece\"}"), true},
      {"a merge naming a symbol the vocabulary lacks", Replace(good, "\"a b\"", "\"a zz\""), false},
      {"a merge whose result is missing", Replace(good, "\"a b\"", "\"a c\""), false},
      {"a merge that is not a pair", Replace(good, "\"a b\"", "[\"a\"]"), false},
      {"a space-joined merge with an empty half", Replace(good, "\"a b\"", "\"a \""), false},
      {"an added token with no id", ByteLevelSpec(",\"added_tokens\":[{\"content\":\"x\"}]"), false},
      {"two added tokens sharing an id", ByteLevelSpec(",\"added_tokens\":[{\"content\":\"x\",\"id\":300},{\"content\":\"y\",\"id\":300}]"), false},
      {"a vocabulary giving one id to two tokens", Replace(good, "\"ab\":256", "\"ab\":0"), false},
      {"no vocabulary", "{\"model\":{\"type\":\"BPE\",\"merges\":[]}}", false},
      {"not JSON", "{", false},
      {"a lossy non-byte-level tokenizer", "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}", true},
  };
  for (const auto& c : cases) {
    auto t = Tokenizer::Parse(c.spec);
    ASSERT_FALSE(t.ok()) << c.name << " must be refused, not loaded";
    EXPECT_EQ(tk::IsUnsupported(t.error()), c.unsupported) << c.name << ": " << t.error();
  }
  auto partial = Tokenizer::Parse("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},\"pre_tokenizer\":{\"type\":\"ByteLevel\"}}");
  ASSERT_FALSE(partial.ok());
  EXPECT_NE(partial.error().find("absent from its vocabulary"), std::string::npos) << partial.error();
}

TEST(TokenizerLoader, UnknownTokenAndFuseUnk) {
  std::string with_unk = "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"<unk>\":1},\"merges\":[],\"unk_token\":\"<unk>\"}}";
  struct Case {
    bool fuse;
    const char* text;
    std::vector<int32_t> want;
  } cases[] = {{false, "azz a", {0, 1, 1, 1, 0}}, {false, "zzz", {1, 1, 1}}, {false, "zz a zz", {1, 1, 1, 0, 1, 1, 1}},
               {true, "azz a", {0, 1, 0}}, {true, "zzz", {1}}, {true, "zz a zz", {1, 0, 1}}};
  for (const auto& c : cases) {
    Tokenizer tok = MustParse(c.fuse ? Replace(with_unk, "\"unk_token\"", "\"fuse_unk\":true,\"unk_token\"") : with_unk);
    EXPECT_EQ(tok.Encode(c.text), c.want) << "fuse=" << c.fuse << " " << c.text;
  }
}

TEST(TokenizerLoader, IdsFarBeyondTheVocabularyAreRefused) {
  for (const char* id : {"2147483647", "1000000000", "70000"}) {
    auto t = Tokenizer::Parse(std::string("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"<unk>\":") + id + "},\"merges\":[],\"unk_token\":\"<unk>\"}}");
    ASSERT_FALSE(t.ok()) << id;
    EXPECT_NE(t.error().find("sparse"), std::string::npos) << t.error();
  }
  EXPECT_TRUE(Tokenizer::Parse("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"<unk>\":9000},\"merges\":[],\"unk_token\":\"<unk>\"}}").ok());
}

TEST(TokenizerAdded, LongestLstripRstripSingleWordNormalized) {
  Tokenizer tok = MustParse(ByteLevelSpec(
      ",\"added_tokens\":[{\"content\":\"<S>\",\"id\":300,\"special\":true,\"normalized\":false},"
      "{\"content\":\"<SS>\",\"id\":301,\"special\":true,\"normalized\":false},"
      "{\"content\":\"[M]\",\"id\":302,\"special\":true,\"lstrip\":true,\"rstrip\":true},"
      "{\"content\":\"word\",\"id\":303,\"single_word\":true},"
      "{\"content\":\"\xC3\xA9" "clair\",\"id\":304,\"normalized\":true},"
      "{\"content\":\"\xC2\xA7\",\"id\":305},{\"content\":\"!\",\"id\":306}],\"decoder\":{\"type\":\"ByteLevel\"}"));
  using V = std::vector<int32_t>;
  EXPECT_EQ(tok.Encode("<SS><S>"), (V{301, 300})) << "the longest token wins at a position";
  EXPECT_EQ(tok.Encode("a   [M]   b"), (V{97, 302, 98})) << "lstrip and rstrip absorb the white space";
  EXPECT_EQ(tok.Encode("word"), (V{303}));
  auto inside = tok.Encode("swords word.");
  ASSERT_GE(inside.size(), 3u);
  EXPECT_EQ(inside[inside.size() - 2], 303) << "a single_word token inside a word is not matched but a free one is";
  EXPECT_EQ(std::count(inside.begin(), inside.end(), 303), 1) << "swords holds the word but not as a word";
  EXPECT_EQ(tok.Encode("\xC3\xA9" "clair"), (V{304}));
  EXPECT_EQ(tok.Encode("a\xC2\xA7" "b!"), (V{97, 305, 98, 306}));
  EXPECT_EQ(tok.Encode("<S"), (V{60, 83})) << "a prefix of a token is not the token";
  EXPECT_EQ(tok.Decode({300, 97, 301}, true), "a");
  EXPECT_EQ(tok.Decode({300, 97}, false), "<S>a");
  EXPECT_EQ(tok.SpecialId("<S>"), std::optional<int32_t>(300));
  EXPECT_FALSE(tok.SpecialId("word").has_value());
  EXPECT_EQ(tok.SpecialTokens().size(), 3u);
}

TEST(TokenizerLoader, RolesFromTokenizerConfigAndDirectoryLayout) {
  std::string base = ::testing::TempDir() + "/laya_tok_roles";
  auto write = [](const std::string& path, const std::string& content) {
    std::ofstream(path, std::ios::binary) << content;
  };
  std::string cmd = "mkdir -p '" + base + "/m/tokenizer' '" + base + "/plain' '" + base + "/empty'";
  ASSERT_EQ(std::system(cmd.c_str()), 0);
  std::string spec = ByteLevelSpec(",\"added_tokens\":[{\"content\":\"<c>\",\"id\":300,\"special\":true},{\"content\":\"<m>\",\"id\":301,\"special\":true}]");
  write(base + "/plain/tokenizer.json", spec);
  write(base + "/plain/tokenizer_config.json", "{\"cls_token\":\"<c>\",\"mask_token\":{\"content\":\"<m>\"},\"pad_token\":null}");
  auto t = Tokenizer::LoadDir(base + "/plain");
  ASSERT_TRUE(t.ok()) << t.error();
  EXPECT_EQ(t.value().Cls(), std::optional<int32_t>(300));
  EXPECT_EQ(t.value().Mask(), std::optional<int32_t>(301));
  EXPECT_EQ(t.value().MaskToken(), "<m>");
  EXPECT_FALSE(t.value().Pad().has_value()) << "a null role is absent";
  EXPECT_FALSE(t.value().Sep().has_value()) << "an unnamed role is absent";
  write(base + "/plain/tokenizer_config.json", "{\"cls_token\":\"<nope>\"}");
  EXPECT_FALSE(Tokenizer::LoadDir(base + "/plain").ok()) << "a role naming a token the vocabulary lacks";
  write(base + "/plain/tokenizer_config.json", "{\"cls_token\":5}");
  EXPECT_FALSE(Tokenizer::LoadDir(base + "/plain").ok()) << "a role that is not a string or an object";
  write(base + "/m/tokenizer/tokenizer.json", ByteLevelSpec(""));
  EXPECT_TRUE(Tokenizer::LoadDir(base + "/m").ok()) << "<dir>/tokenizer/ is preferred";
  EXPECT_FALSE(Tokenizer::LoadDir(base + "/empty").ok()) << "a directory with no tokenizer";
}

TEST(TokenizerRefusals, NormalizersPreTokenizersAndRegexDialects) {
  for (const char* spec : {"{\"type\":\"BertNormalizer\"}", "{}", "{\"type\":\"Replace\",\"pattern\":{\"String\":\"a\"}}",
                           "{\"type\":\"Replace\",\"pattern\":{\"String\":\"\"},\"content\":\"x\"}",
                           "{\"type\":\"Replace\",\"pattern\":{},\"content\":\"x\"}",
                           "{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"(\"},\"content\":\"x\"}", "{\"type\":\"Sequence\"}",
                           "{\"type\":\"Sequence\",\"normalizers\":[1]}"}) {
    EXPECT_FALSE(tk::RunNormalizerSpec(spec, "a").ok()) << spec;
  }
  // The reference reads \s, \d, \w, \b, \p as Unicode; an ASCII engine would differ, so they are refused.
  for (const char* expr : {"\\\\s+", "a\\\\bb", "[\\\\d]", "\\\\w", "\\\\p{L}"}) {
    auto r = tk::RunNormalizerSpec(std::string("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"") + expr + "\"},\"content\":\"x\"}", "a");
    ASSERT_FALSE(r.ok()) << expr;
    EXPECT_TRUE(tk::IsUnsupported(r.error())) << expr << ": " << r.error();
  }
  auto esc = tk::RunNormalizerSpec("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\\\\\\\\\\\\\s\"},\"content\":\"x\"}", "a\\\\s");
  ASSERT_TRUE(esc.ok()) << esc.error();
  EXPECT_EQ(esc.value(), "ax") << "an escaped backslash before s is a literal, not a class";
  auto empty = tk::RunNormalizerSpec("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"a*\"},\"content\":\"x\"}", "b");
  ASSERT_FALSE(empty.ok());
  EXPECT_TRUE(tk::IsUnsupported(empty.error())) << "a pattern that can match nothing";
  for (const char* spec : {"{\"type\":\"Whitespace\"}", "{}", "{\"type\":\"Metaspace\",\"replacement\":\"ab\"}",
                           "{\"type\":\"Metaspace\",\"prepend_scheme\":\"sometimes\"}",
                           "{\"type\":\"Sequence\",\"pretokenizers\":[1]}"}) {
    EXPECT_FALSE(tk::RunPreTokenizerSpec(spec, "a").ok()) << spec;
  }
  EXPECT_TRUE(tk::IsUnsupported(tk::RunPreTokenizerSpec("{\"type\":\"Whitespace\"}", "a").error()));
}

TEST(TokenizerRegex, SubsetBehaviour) {
  auto run = [](const std::string& pattern, const std::string& in, const std::string& to = "_") {
    auto r = tk::RunNormalizerSpec("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"" + pattern + "\"},\"content\":\"" + to + "\"}", in);
    EXPECT_TRUE(r.ok()) << pattern << ": " << r.error();
    return r.ok() ? r.value() : std::string("<error>");
  };
  EXPECT_EQ(run("a|b", "cabd"), "c__d");
  EXPECT_EQ(run("a+?", "aaa"), "___") << "lazy repeat";
  EXPECT_EQ(run("a+", "baaac"), "b_c") << "greedy repeat";
  EXPECT_EQ(run("(ab)+c", "ababc abc"), "_ _");
  EXPECT_EQ(run("[^a-c]", "abxcy"), "ab_c_");
  EXPECT_EQ(run("^a", "aa"), "_a");
  EXPECT_EQ(run("a$", "aa"), "a_");
  EXPECT_EQ(run("^a", "a\na"), "_\n_") << "^ and $ are line anchors, as in the crate's Oniguruma";
  EXPECT_EQ(run("a$", "a\na"), "_\n_");
  EXPECT_EQ(run("(a?)?b", "ab"), "_");
  auto nullable = tk::RunNormalizerSpec(R"({"type":"Replace","pattern":{"Regex":"(a*)+b"},"content":"x"})", "b");
  ASSERT_FALSE(nullable.ok());
  EXPECT_TRUE(tk::IsUnsupported(nullable.error())) << "repeating a group that can match nothing";
  EXPECT_EQ(run("a.c", "abc a\\nc"), "_ a\\nc");
  EXPECT_EQ(run("x{2,3}", "xxxxxxx"), "__x");
  EXPECT_EQ(run("(?i)HeLLo", "Hello HELLO hello"), "_ _ _");
  EXPECT_EQ(run("é+", "ééa"), "_a");
  EXPECT_EQ(run("(a|ab)(c|bcd)", "abcd"), "_") << "leftmost-first: a then bcd";
  // The failed-state memo keeps repeats linear.
  EXPECT_EQ(run("(a|aa)+b", std::string(5000, 'a')), std::string(5000, 'a'));
}

TEST(TokenizerPre, SequenceAndCornerCases) {
  auto seq = tk::RunPreTokenizerSpec(
      "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Metaspace\",\"prepend_scheme\":\"never\"},{\"type\":\"ByteLevel\",\"add_prefix_space\":false}]}", "ab cd");
  ASSERT_TRUE(seq.ok());
  EXPECT_EQ(Join(seq.value()), Join({"ab", "\xC3\xA2\xC4\xB8\xC4\xA3", "cd"})) << "the crate gives ['ab', 'âĸģ', 'cd']";
  const char* spec = "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true}";
  struct Case {
    const char* in;
    std::vector<std::string> want;
  } cases[] = {
      {"a  b\t\tc\n\nd\r\ne   f", {"a", "\xC4\xA0", "\xC4\xA0" "b", "\xC4\x89", "\xC4\x89", "c", "\xC4\x8A", "\xC4\x8A", "d", "\xC4\x8D", "\xC4\x8A", "e", "\xC4\xA0\xC4\xA0", "\xC4\xA0" "f"}},
      {"abc123", {"abc", "123"}}, {"123abc", {"123", "abc"}}, {"a 1 b", {"a", "\xC4\xA0" "1", "\xC4\xA0" "b"}},
      {"x's value", {"x", "'s", "\xC4\xA0" "value"}}, {"", {}}};
  for (const auto& c : cases) {
    auto got = tk::RunPreTokenizerSpec(spec, c.in);
    ASSERT_TRUE(got.ok());
    EXPECT_EQ(Join(got.value()), Join(c.want)) << c.in;
  }
}

// Short pieces merge by rescanning and long ones with a heap; both must give one answer on real
// text, including pieces long enough for the heap.
TEST_P(Shipped, HeapAndRescanMergeAgree) {
  std::string name = GetParam();
  LAYA_GOLDEN_LIST(side, "go_extra.json", "encode_fuzz", name.c_str());
  LAYA_CHECKPOINT(tok, name);
  size_t checked = 0, bad = 0;
  for (const Value& t : side->Find("texts")->items()) {
    bad += tok->CountMergeDisagreements(t.as_string(), &checked);
    std::string longer;  // a long piece: the heap path is taken from 32 symbols
    for (char c : t.as_string()) {
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) longer += c;
    }
    while (!longer.empty() && longer.size() < 200) longer += longer;
    bad += tok->CountMergeDisagreements(longer, &checked);
  }
  EXPECT_EQ(bad, 0u);
  EXPECT_GE(checked, 1000u);
}

TEST(TokenizerBpe, DuplicateMergeKeepsTheLowestRank) {
  // "a b" is listed at rank 0 and again at rank 2, after "b c": the first listing decides, so abc -> ab, c.
  Tokenizer tok = MustParse(
      R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"<unk>":5},"merges":["a b","b c","a b"],"unk_token":"<unk>"}})");
  EXPECT_EQ(tok.Encode("abc"), (std::vector<int32_t>{3, 2}));
}

TEST(TokenizerBpe, ByteFallbackNeedsEveryByteAndResetsFusedUnknowns) {
  const std::string head = R"({"model":{"type":"BPE","vocab":{"a":0,"<unk>":1,"<0xC3>":2,"<0xA9>":3},"merges":[],"unk_token":"<unk>","byte_fallback":true)";
  using V = std::vector<int32_t>;
  Tokenizer ok = MustParse(head + "}}");
  EXPECT_EQ(ok.Encode("\xC3\xA9"), (V{2, 3})) << "every byte of the character resolves";
  EXPECT_EQ(ok.Encode("\xC3\x80"), (V{1})) << "<0x80> is missing, so the whole character is unknown, not half bytes";
  Tokenizer fused = MustParse(head + ",\"fuse_unk\":true}}");
  EXPECT_EQ(fused.Encode("z\xC3\xA9z"), (V{1, 2, 3, 1})) << "bytes between two unknowns end the fused run";
  EXPECT_EQ(fused.Encode("zzz"), (V{1}));
}

TEST(TokenizerRefusals, SequenceRunsInOrder) {
  auto r = tk::RunNormalizerSpec(
      R"({"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"a"},{"type":"Replace","pattern":{"String":"a"},"content":"b"}]})", "a");
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value(), "bb");
  auto s = tk::RunNormalizerSpec(
      R"({"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"String":"a"},"content":"b"},{"type":"Prepend","prepend":"a"}]})", "a");
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(s.value(), "ab");
}

TEST(TokenizerAdded, RstripAbsorbsAllWhiteSpace) {
  Tokenizer tok = MustParse(ByteLevelSpec(",\"added_tokens\":[{\"content\":\"[M]\",\"id\":302,\"special\":true,\"rstrip\":true}]"));
  EXPECT_EQ(tok.Encode("[M]  \t b"), (std::vector<int32_t>{302, 98}));
}

// Canonical ordering must run before composition: a + dot above (230) + dot below (220) is a-dot-below
// then dot above (values recorded from the crate).
TEST(TokenizerNormalize, MarksAreReorderedBeforeTheyCompose) {
  EXPECT_EQ(tk::Normalize("a\xCC\x87\xCC\xA3", Form::NFC), "\xE1\xBA\xA1\xCC\x87");
  EXPECT_EQ(tk::Normalize("a\xCC\x87\xCC\xA3", Form::NFKC), "\xE1\xBA\xA1\xCC\x87");
  EXPECT_EQ(tk::Normalize("a\xCC\x87\xCC\xA3", Form::NFD), "a\xCC\xA3\xCC\x87");
}
