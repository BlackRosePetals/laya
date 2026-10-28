#pragma once
// The ONNX Runtime backend (laya-go/onnx). Linking laya_onnx pulls in ONNX Runtime; the core `laya`
// library never does, and this header includes no ONNX Runtime header.
#include <cstdint>
#include <memory>
#include <string>
#include "laya/agent.hpp"
#include "laya/infer.hpp"
#include "laya/result.hpp"

namespace laya {
namespace onnx {

struct Options {
  int intra_op_threads = 0;  // 0 leaves ONNX Runtime's default; at most kMaxThreads
  int inter_op_threads = 0;
};
constexpr int kMaxThreads = 4096;

// The version of the ONNX Runtime library loaded, e.g. "1.30.0".
std::string RuntimeVersion();

// A Session over one fused laya.onnx (+ laya.onnx.data beside it). Checks the graph's inputs and
// outputs at once. Run is thread-safe; Close is idempotent and waits for runs in flight. No
// exception leaves this API.
Result<std::unique_ptr<infer::Session>> Open(const std::string& graph_path, const Options& options = {});

}  // namespace onnx

namespace agent {
// checkpoint_dir holds rl_agent_config.json and the tokenizer; graph_path is the fused laya.onnx.
Result<std::unique_ptr<Agent>> OpenOnnx(const std::string& checkpoint_dir, const std::string& graph_path,
                                        const onnx::Options& options = {});
}  // namespace agent

}  // namespace laya
