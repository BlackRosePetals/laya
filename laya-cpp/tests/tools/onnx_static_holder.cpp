// Holds a Session and an Agent in statics, as a user program may, then exits normally. Exit code 0
// means static destruction did not fault. argv: graph checkpoint_dir [close].
#include <cstdio>
#include <cstring>
#include <memory>

#include "laya/onnx.hpp"

namespace {
std::unique_ptr<laya::infer::Session> g_session;  // constructed before any ONNX Runtime static
std::unique_ptr<laya::agent::Agent> g_agent;
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 64;
  auto s = laya::onnx::Open(argv[1]);
  if (!s.ok()) {
    std::fprintf(stderr, "open: %s\n", s.error().c_str());
    return 2;
  }
  g_session = std::move(s).value();
  auto a = laya::agent::OpenOnnx(argv[2], argv[1]);
  if (!a.ok()) {
    std::fprintf(stderr, "agent: %s\n", a.error().c_str());
    return 3;
  }
  g_agent = std::move(a).value();
  if (argc > 3 && !std::strcmp(argv[3], "close")) {
    if (!g_session->Close().empty() || !g_agent->Close().empty()) return 4;
  }
  return 0;
}
