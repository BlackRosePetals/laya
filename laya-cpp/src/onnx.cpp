#include "laya/onnx.hpp"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "laya/onnx_graph.hpp"

namespace laya {
namespace onnx {
namespace {

// One Ort::Env per process, shared by every session and freed with the last one, so there is no
// static destructor racing a session's teardown. The mutex orders creation against that release.
std::shared_ptr<Ort::Env> SharedEnv() {
  static std::mutex mu;
  static std::weak_ptr<Ort::Env> weak;
  std::lock_guard<std::mutex> lock(mu);
  std::shared_ptr<Ort::Env> env = weak.lock();
  if (!env) {
    env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "laya");
    weak = env;
  }
  return env;
}

std::string Dtype(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    default: return "type " + std::to_string(int(t));
  }
}

// Inputs by name, in the order the five tensors are bound below.
const char* const kInputNames[5] = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
const char* const kOutputNames[2] = {"logits", "act_logits"};

class OrtSession final : public infer::Session {
 public:
  OrtSession(std::shared_ptr<Ort::Env> env, std::unique_ptr<Ort::Session> s) : env_(std::move(env)), session_(std::move(s)) {}
  ~OrtSession() override { Close(); }

  Result<infer::Output> Run(const infer::Batch& b, const CancelToken& cancel) override {
    try {
      return RunImpl(b, cancel);
    } catch (const Ort::Exception& e) {
      return Result<infer::Output>::Fail(std::string("onnx: run: ") + e.what());
    } catch (const std::exception& e) {
      return Result<infer::Output>::Fail(std::string("onnx: run: ") + e.what());
    } catch (...) {
      return Result<infer::Output>::Fail("onnx: run: unknown exception");
    }
  }

  std::string Close() override {
    try {
      std::unique_lock<std::shared_mutex> lock(mu_);  // waits for runs in flight
      session_.reset();
      return "";
    } catch (...) {
      return "onnx: close failed";
    }
  }

 private:
  Result<infer::Output> RunImpl(const infer::Batch& b, const CancelToken& cancel) {
    using R = Result<infer::Output>;
    if (cancel.cancelled()) return R::Fail(infer::kCancelled);
    const std::string bad = b.Validate();
    if (!bad.empty()) return R::Fail("onnx: " + bad);
    for (uint8_t m : b.marker_mask) {
      if (m > 1) return R::Fail("onnx: marker_mask holds a value other than 0 or 1");
    }
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!session_) return R::Fail("onnx: session is closed");

    const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape2[2] = {b.rows, b.length};
    const int64_t shape_m[2] = {b.rows, b.markers};
    const int64_t shape1[1] = {b.rows};
    // ORT takes non-const pointers but never writes to an input.
    auto i64 = [&](const std::vector<int64_t>& v, const int64_t* shape, size_t rank) {
      return Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t*>(v.data()), v.size(), shape, rank);
    };
    std::vector<Ort::Value> in;
    in.reserve(5);
    in.push_back(i64(b.input_ids, shape2, 2));
    in.push_back(i64(b.attention_mask, shape2, 2));
    in.push_back(i64(b.marker_pos, shape_m, 2));
    in.push_back(Ort::Value::CreateTensor(mem, const_cast<uint8_t*>(b.marker_mask.data()), b.marker_mask.size() * sizeof(uint8_t),
                                          shape_m, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
    in.push_back(i64(b.qtype, shape1, 1));

    Ort::RunOptions ro;
    std::vector<Ort::Value> out = session_->Run(ro, kInputNames, in.data(), in.size(), kOutputNames, 2);
    if (out.size() != 2) return R::Fail("onnx: run returned " + std::to_string(out.size()) + " outputs, want 2");
    if (cancel.cancelled()) return R::Fail(infer::kCancelled);

    OutTensor t[2];
    for (size_t i = 0; i < 2; i++) {
      if (!out[i].IsTensor()) return R::Fail(std::string("onnx: output ") + kOutputNames[i] + " is not a tensor");
      const Ort::TensorTypeAndShapeInfo info = out[i].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return R::Fail(std::string("onnx: output ") + kOutputNames[i] + " is " + Dtype(info.GetElementType()) + ", want float32");
      }
      t[i].shape = info.GetShape();
      t[i].count = int64_t(info.GetElementCount());
      t[i].data = out[i].GetTensorData<float>();
    }
    return CollectOutput(b, t[0], t[1]);
  }

  std::shared_ptr<Ort::Env> env_;  // keeps the environment alive past session_
  std::shared_mutex mu_;
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace

std::string RuntimeVersion() {
  try {
    return Ort::GetVersionString();
  } catch (...) {
    return "";
  }
}

Result<std::unique_ptr<infer::Session>> Open(const std::string& graph_path, const Options& options) {
  using R = Result<std::unique_ptr<infer::Session>>;
  if (options.intra_op_threads < 0 || options.inter_op_threads < 0 || options.intra_op_threads > kMaxThreads ||
      options.inter_op_threads > kMaxThreads) {
    return R::Fail("onnx: thread counts must be between 0 and " + std::to_string(kMaxThreads));
  }
  const std::string path_problem = CheckPath(graph_path);
  if (!path_problem.empty()) return R::Fail("onnx: " + path_problem);
  try {
    std::shared_ptr<Ort::Env> env = SharedEnv();
    Ort::SessionOptions so;
    if (options.intra_op_threads > 0) so.SetIntraOpNumThreads(options.intra_op_threads);
    if (options.inter_op_threads > 0) so.SetInterOpNumThreads(options.inter_op_threads);
    const std::filesystem::path p(graph_path);
    auto session = std::make_unique<Ort::Session>(*env, p.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    auto describe = [&](bool inputs) {
      std::vector<TensorInfo> v;
      const size_t n = inputs ? session->GetInputCount() : session->GetOutputCount();
      for (size_t i = 0; i < n; i++) {
        TensorInfo t;
        Ort::AllocatedStringPtr name = inputs ? session->GetInputNameAllocated(i, alloc) : session->GetOutputNameAllocated(i, alloc);
        t.name = name.get();
        Ort::TypeInfo ti = inputs ? session->GetInputTypeInfo(i) : session->GetOutputTypeInfo(i);
        if (ti.GetONNXType() != ONNX_TYPE_TENSOR) {
          t.dtype = "not a tensor";
        } else {
          const Ort::ConstTensorTypeAndShapeInfo info = ti.GetTensorTypeAndShapeInfo();
          t.dtype = Dtype(info.GetElementType());
          t.dims = info.GetShape();
        }
        v.push_back(std::move(t));
      }
      return v;
    };
    const std::string bad = CheckGraph(describe(true), describe(false));
    if (!bad.empty()) return R::Fail("onnx: graph " + graph_path + ": " + bad);
    return R::Ok(std::unique_ptr<infer::Session>(new OrtSession(std::move(env), std::move(session))));
  } catch (const Ort::Exception& e) {
    return R::Fail("onnx: cannot open " + graph_path + ": " + e.what());
  } catch (const std::exception& e) {
    return R::Fail("onnx: cannot open " + graph_path + ": " + e.what());
  } catch (...) {
    return R::Fail("onnx: cannot open " + graph_path + ": unknown exception");
  }
}

}  // namespace onnx

namespace agent {
Result<std::unique_ptr<Agent>> OpenOnnx(const std::string& checkpoint_dir, const std::string& graph_path, const onnx::Options& options) {
  auto s = onnx::Open(graph_path, options);
  if (!s.ok()) return Result<std::unique_ptr<Agent>>::Fail(s.error());
  return Agent::Open(checkpoint_dir, std::move(s).value());  // closes the session on failure
}
}  // namespace agent

}  // namespace laya
