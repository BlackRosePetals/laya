#pragma once
// A cancellation token. Copies share one flag, so the caller keeps a copy and cancels it from any
// thread while a call holds another. A default token can never be cancelled.
#include <atomic>
#include <memory>

namespace laya {

class CancelToken {
 public:
  CancelToken() = default;
  static CancelToken Make() {
    CancelToken t;
    t.flag_ = std::make_shared<std::atomic<bool>>(false);
    return t;
  }
  void Cancel() const {
    if (flag_) flag_->store(true, std::memory_order_release);
  }
  bool cancelled() const { return flag_ && flag_->load(std::memory_order_acquire); }

 private:
  std::shared_ptr<std::atomic<bool>> flag_;
};

}  // namespace laya
