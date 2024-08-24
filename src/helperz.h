#pragma once
#include <torch/csrc/autograd/python_variable.h>
#include <random>

#include "neuralBase.h"
// #define _unused(x) ((void)(x))
// #include <einops.hpp>

static constexpr long pModulo { 113 }, DATA_SEED { 598 };
static constexpr float frac_train { 0.3f }, lr { 1e-3f }, wd { 1.f }, betas[] { 0.9f, 0.98f };
static constexpr long num_epochs { 25000 }, checkpoint_every { 100 }, numTotal { pModulo * pModulo };
static constexpr long numTrain { static_cast<long>(numTotal * frac_train) }, numTest { numTotal - numTrain };
template<typename T> T pi = std::numbers::pi_v<T>;
static inline c10::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
using namespace torch::indexing;
using TensorMap = std::map<std::string, at::Tensor>;

namespace einsad {
  using NamedInt = std::tuple<std::string, int64_t>;
  template<class... Axes> at::Tensor rearrange(at::Tensor x, std::string pattern, Axes... a);
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a);
  template<class... Ts> at::Tensor einsum(std::string pattern, Ts... ts);
}

static inline at::Tensor getMode(at::Tensor all, const std::string& mode) {
  return mode == "train" ? all.slice(0, 0, numTrain) : mode == "test" ? all.slice(0, numTrain, numTotal) : all; }

static inline TensorMap makeINDICES() { print("in makeINDICES()");
  TensorMap indices;
  torch::manual_seed(uint64_t(DATA_SEED));
  indices["all"] = torch::randperm(pModulo*pModulo);
  indices["train"] = getMode(indices["all"], "train");
  indices["test"] = getMode(indices["all"], "test");
  return indices;
}

static inline TensorMap indices { makeINDICES() };

struct ActivationCache {
  struct ActKeys {
    template<typename T> std::string stringify(const T& x) { if constexpr (std::is_integral_v<T>) return std::to_string(int(x)); else return { x }; }
    template<typename ...Ts> ActKeys(Ts... ts) { forEach([&](auto x){ data.push_back(stringify(x)); }, ts...); } 
    std::vector<std::string> data;
  };
  // struct ActKeys {
  //   ActKeys(const std::string& l, int i, const std::string& r) { data.push_back(l); data.push_back(std::to_string(i)); data.push_back(r); }
  //   ActKeys(const std::string& l, int i) { data.push_back(l); data.push_back(std::to_string(i)); }
  //   ActKeys(const std::string& l) { data.push_back(l); }
  //   std::vector<std::string> data;
  // };
  auto begin() { return data.begin(); }
  const auto begin() const { return data.begin(); }
  auto end() { return data.end(); }
  const auto end() const { return data.end(); }
  at::Tensor operator[](ActKeys keys) {
    for(auto& [n, t] : data) { 
      bool found = true; 
      for(const auto& x : keys.data)
        if (n.find(x) == std::string::npos) 
          found = false;
      if (found) return t;
    }
    for(const auto& [n, t] : data) print(n, t.sizes());
    print("ERROR - keys not in ActivationCache!!! Returning Tensor()", keys.data, "cache size", data.size()); 
    return torch::Tensor();
  }
  // at::Tensor operator[](const std::tuple<std::string, int>& keys) { const auto& [a,b] = keys; return operator[]({ a,b,a }); }
  bool contains(const std::string& s) const {
    for(const auto& [name, tensor] : data)
      if (name == s) return true;
    return false;
  }
  std::vector<std::pair<std::string, at::Tensor>> data;
};

template<typename T> T cross_entropy_high_precision(T x, T y) { return getCrossEntropy(x.to(torch::kFloat64), y); }
template<typename T> T test_logits(T logits, T labels) { return loss_fn(logits, labels); }
template<typename T> T test_logits(T logits, T labels, bool bias_correction, T original_logits) {
  if (bias_correction) logits = (original_logits - logits).mean(1, true) + logits;
  return loss_fn(logits, labels);
}

// at::Tensor test_logits(torch::Tensor logits, const std::string& mode, bool bias_correction = false, torch::Tensor* original_logits = nullptr) {
//   return test_logits(logits, LABELS[mode], bias_correction, original_logits);
// }

at::Tensor unflatten_first(at::Tensor x);

static constexpr long key_freqs[] = { 17, 25, 32, 47 }, key_freq_indices[] = { 33, 34, 49, 50, 63, 64, 93, 94 };
static constexpr auto num_key_freqs = sizeof(key_freqs)/sizeof(key_freqs[0]);

static inline std::vector<std::string> makeFourierBasisNames() {
  std::vector<std::string> names { "Constant" };
  for (long freq { 1 }; freq < pModulo/2 + 1; ++freq) {
    names.push_back("cos " + std::to_string(freq));
    names.push_back("sin " + std::to_string(freq));
  }
  return names;
}

static inline std::vector<std::string> fourier_basis_names { makeFourierBasisNames() }; // VARIABLE

struct Tester {

        Tester() { makeTrigData(); }

        void makeTrigData() {
          constexpr long p { pModulo };
          auto a = torch::arange(p).unsqueeze(1).to(device);//.index({ Slice(), None });
          auto b = torch::arange(p).unsqueeze(0).to(device);//.index({ None, Slice() });
          for (const auto& freq : key_freqs) {
            auto cos_vec = torch::cos((freq * 2.f * pi<float> / p) * (a + b));
            cos_vec = cos_vec/cos_vec.norm();
            cos_vec = einsad::rearrange(cos_vec, "a b -> (a b) 1").index({ indices["all"] });
            auto sin_vec = torch::sin((freq * 2.f * pi<float> / p) * (a + b));
            sin_vec = sin_vec/sin_vec.norm();
            sin_vec = einsad::rearrange(sin_vec, "a b -> (a b) 1").index({ indices["all"] });
            cos_apb_map[freq] = cos_vec; 
            sin_apb_map[freq] = sin_vec;
          }         
        }

        static at::Tensor getLastLogit(at::Tensor x) { return x.dim() > 2 ? x.slice(1, x.size(1) - 1, x.size(1)).squeeze(1) : x; }//.slice(-1, 0, x.size(-1) - 1) }

        TensorMap splitModes(at::Tensor x) {
          TensorMap modeMap;
          // modeMap["all"] = x;
          modeMap["train"] = getMode(x, "train");
          modeMap["test"] = getMode(x, "test");
          return modeMap;
        }

        TensorMap getLogitMetrics(at::Tensor logits, TensorMap& LABELS, const std::string& name, at::Tensor origLogits = at::Tensor()) {
          // print("getLogitMetrics name, logits", name, logits.sizes(), get_first_elements(logits, 3));
          auto logit_modes = splitModes(logits);
      
          TensorMap retMap;
          for (auto& [mode, tensor] : logit_modes) {
            auto labels { LABELS[mode].to(tensor.device()) };

            // print("mode, tensor", mode, tensor.sizes(), get_first_elements(tensor, 3));
            retMap[mode + (name.empty() ? "" : "_" + name) + "_loss"] = cross_entropy_high_precision(tensor, labels);
            retMap[mode + (name.empty() ? "" : "_" + name) + "_accuracy"] = getAccuracy(tensor, labels);
          }
          return retMap;
        }

        TensorMap getRestrictedLoss(at::Tensor logits, TensorMap LABELS, at::Tensor W_out, at::Tensor W_U, ActivationCache& cache) {
          // print("logits", logits.sizes(), get_first_elements(logits, 3));
          logits = getLastLogit(logits);
          // print("logits", logits.sizes(), get_first_elements(logits, 3));
          auto neuron_acts { cache[{ "post", 0, "mlp" }].index({ Slice(), -1, Slice() }) };
          auto resid_mid { cache[{ "resid_mid", 0 }].index({ Slice(), -1, Slice() }) };
          auto approx_neuron_acts = torch::zeros_like(neuron_acts);
          approx_neuron_acts += neuron_acts.mean(0);

          for (auto freq : key_freqs) {
            approx_neuron_acts += (neuron_acts * cos_apb_map[freq]).sum(0) * cos_apb_map[freq];
            approx_neuron_acts += (neuron_acts * sin_apb_map[freq]).sum(0) * sin_apb_map[freq];
          }
          auto restricted_logits = torch::matmul(torch::matmul(approx_neuron_acts, W_out), W_U);
          restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);

          auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
          auto residual_stream_final = torch::matmul(excluded_neuron_acts, W_out) + resid_mid;
          auto excluded_logits = torch::matmul(residual_stream_final, W_U);
          auto lossAndAccuracy { getLogitMetrics(logits, LABELS, "") };
          auto restricted { getLogitMetrics(restricted_logits, LABELS, "restricted") };
          lossAndAccuracy.merge(restricted);
          auto excluded { getLogitMetrics(excluded_logits, LABELS, "excluded") };
          lossAndAccuracy.merge(excluded);
          return lossAndAccuracy;
        }

        TensorMap getLoss(at::Tensor logits, TensorMap& LABELS) { return getLogitMetrics(logits, LABELS, ""); }

        at::Tensor makeFourierBasis() {
          constexpr long p { pModulo };
          std::vector<at::Tensor> basis { torch::ones(p)/std::sqrt(float(p)) }; // starts with a DC (constant) vector
          for (long freq { 1 }; freq < p/2 + 1; ++freq) { // starting at freq=1 add a pair of basis vectors cos(freq) sin(freq).. to p/2 (56)
            basis.push_back(torch::cos(torch::arange(p) * (2.f * pi<float> * (float)freq / (float)p)));
            basis.push_back(torch::sin(torch::arange(p) * (2.f * pi<float> * (float)freq / (float)p)));
            // basis[basis.size() - 2] /= basis[basis.size() - 2].norm();
            // basis[basis.size() - 1] /= basis[basis.size() - 1].norm();
          }
          auto fbasis = torch::stack(basis, 0L);
          fbasis = fbasis/torch::square(fbasis).sum(1, true).sqrt();
          print("fourier basis sizes, isnan", fbasis.sizes(), fbasis.isnan().any());
          return fbasis;
        }

        void checknan(torch::Tensor x, std::string name = "") {
          // print(x.isnan().any());
          if (x.isnan().any().item<bool>() == true)
            print("found nan " + name);
        }

        template <int N = 0> at::Tensor fourier_2d_basis_term(long x_index, long y_index) {
          if (x_index < 1 || y_index < 1) {
            print("index less than 1", x_index, y_index);
            std::terminate();
          }
          auto i0 = fourier_basis[x_index];
          checknan(i0, "i0");
          auto i1 = i0.index({ Slice(), None });
          checknan(i1, "i1");
          auto i2 = fourier_basis[y_index].index({ None, Slice() });
          checknan(i2, "i2");
          auto i3 = i1 * i2;
          // print("i1", i1.sizes(), get_first_elements(i1, 3));
          // print("i2", i2.sizes(), get_first_elements(i2, 3));
          // print("i3", i3.sizes(), get_first_elements(i3, 3));
          checknan(i3, "i3");
          return fourier_basis[x_index].index({ Slice(), None }) * fourier_basis[y_index].index({ None, Slice() }).flatten();
        }

        template <int N = 0> at::Tensor cos_xpy(at::Tensor tensor, long freq, bool collapse_dim = false) {
          tensor = tensor.to(torch::kCPU);
          auto cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten().to(tensor.device());
          auto sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten().to(tensor.device());
          // print("cos cosx_cosy_direction isnan freq, 2*freq-1, 2*freq", freq, 2*freq-1, 2*freq, cosx_cosy_direction.isnan().any());
          // print("cos sinx_siny_direction isnan", sinx_siny_direction.isnan().any());
          auto cos_xpy_dir = (cosx_cosy_direction - sinx_siny_direction)/std::sqrt(2.f);
          if (tensor.size(0) ==  numTrain) cos_xpy_dir = cos_xpy_dir.slice(0, 0, numTrain);
          else if (tensor.size(0) ==  numTest) cos_xpy_dir = cos_xpy_dir.slice(0, numTrain, numTotal);
          auto cos1 = cos_xpy_dir.index({ Slice(), None }), cos2 = cos_xpy_dir.index({ None, Slice() });
          // print ("cos1, cos2, tensor", cos1.sizes(), cos2.sizes(), tensor.sizes());
          // print("cos_xpy_dir isnan", cos_xpy_dir.isnan().any());
          // print("cos tensor isnan", tensor.isnan().any());
          if (collapse_dim) return torch::matmul(cos_xpy_dir, tensor);
          else return torch::matmul(torch::matmul(cos1, cos2), tensor); 
        }
        
        template <int N = 0> at::Tensor sin_xpy(at::Tensor tensor, long freq, bool collapse_dim = false) {
          tensor = tensor.to(torch::kCPU);
          auto sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten().to(tensor.device());
          auto cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten().to(tensor.device());
          // print("sin sinx_cosy_direction isnan freq, 2*freq-1, 2*freq", freq, 2*freq-1, 2*freq, sinx_cosy_direction.isnan().any());
          // print("sin cosx_siny_direction isnan", cosx_siny_direction.isnan().any());
          auto sin_xpy_dir = (sinx_cosy_direction + cosx_siny_direction)/std::sqrt(2.f);
          if (tensor.size(0) ==  numTrain) sin_xpy_dir = sin_xpy_dir.slice(0, 0, numTrain);
          else if (tensor.size(0) ==  numTest) sin_xpy_dir = sin_xpy_dir.slice(0, numTrain, numTotal);
          auto sin1 = sin_xpy_dir.index({ Slice(), None }), sin2 = sin_xpy_dir.index({ None, Slice() });
          // print("sin_xpy_dir isnan", sin_xpy_dir.isnan().any());
          // print("sin tensor isnan", tensor.isnan().any());
          if (collapse_dim) return torch::matmul(sin_xpy_dir, tensor);
          // else return torch::matmul(torch::matmul(sin_xpy_dir.index({ Slice(), None }), sin_xpy_dir.index({ None, Slice() })), tensor); 
          else return torch::matmul(torch::matmul(sin1, sin2), tensor); 
        }

        at::Tensor fourier_basis { makeFourierBasis().to(torch::kCPU) }; // VARIABLE
        
        // at::Tensor cos_cube;// { makeCosCube() }; // VARIABLE
        at::Tensor cos_apb_vec, sin_apb_vec; // VARIABLE
        std::map<int, at::Tensor> cos_apb_map, sin_apb_map;
};

