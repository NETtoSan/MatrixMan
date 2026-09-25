#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

namespace {

class MatrixManHooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::Device getDeviceFromPtr(void*) const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }
};

class MatrixManGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device device) const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}

  c10::Stream getStream(c10::Device device) const override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream getNewStream(c10::Device device, int) const override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream stream) const noexcept override {
    return stream;
  }

  c10::DeviceIndex deviceCount() const noexcept override { return 1; }

  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, MatrixManGuardImpl);

MatrixManHooks hooks;
bool registered = false;

void register_matrixman_hooks() {
  if (!registered) {
    at::RegisterPrivateUse1HooksInterface(&hooks);
    registered = true;
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_hooks", &register_matrixman_hooks);
}
