#pragma once

#include "helperz.h"
#include "math.h"

using hi_res = std::chrono::high_resolution_clock;
struct AllData { TensorMap dataset, labels; };

static inline AllData makeAllData(c10::Device d) {
  print("in makeAllData()");
  constexpr long p { pModulo };

  AllData allData;
  auto& [dataset, labels] = allData;

  auto a { torch::arange(p).unsqueeze(1).repeat({ 1, p }).flatten() };
  auto b { torch::arange(p).unsqueeze(0).repeat({ p, 1 }).flatten() };
  auto eq { torch::full({ p * p }, p) };

  dataset["all"] = torch::stack({ a, b, eq }, 1).to(device);
  labels["all"] = ((a + b) % p).to(device);
  dataset["all"] = dataset["all"].index({ indices["all"] });
  labels["all"] = labels["all"].index({ indices["all"] });

  dataset["train"] = dataset["all"].slice(0, 0, numTrain);
  dataset["test"] = dataset["all"].slice(0, numTrain, numTotal);
  labels["train"] = labels["all"].slice(0, 0, numTrain);
  labels["test"] = labels["all"].slice(0, numTrain, numTotal);
  
  return allData;
}

struct IArchitecture { virtual void reset() = 0; virtual std::shared_ptr<IArchitecture> clone() = 0; };

template<class C>
struct IArchitectureClonableX : public torch::nn::Cloneable<C>, IArchitecture {
	virtual std::shared_ptr<IArchitecture> clone() override {
		return std::dynamic_pointer_cast<IArchitecture>(torch::nn::Cloneable<C>::clone());
	}
};

template <typename T> using IArchitectureClonable = torch::nn::Cloneable<T>;

template<template <int x,int y> class ModuleTemplate, typename ConfigType = GPTConfig, int N = 1, int n = 0> 
struct ModuleStack : torch::nn::Module {// torch::nn::Cloneable<ModuleStack<ModuleTemplate, ConfigType, N, n>> {
  using ModuleTypeImpl = ModuleTemplate<N,n>;
  using SubModuleTypeImpl = ModuleStack<ModuleTemplate, ConfigType, N, n + 1>;
  TORCH_MODULE(ModuleType);
  TORCH_MODULE(SubModuleType);
  ConfigType c;
  ModuleType module;
  SubModuleType subModules;
  template<typename ...Ts> ModuleStack(const ConfigType cfg, Ts ...ts) : c(cfg),
  module(this->register_module(std::to_string(n), ModuleType(cfg, ts...))),
  subModules(this, c, ts...) {}
  template<typename ...Ts> ModuleStack(torch::nn::Module* parent, const ConfigType cfg, Ts ...ts) : c(cfg),
  module(parent->register_module(std::to_string(n), ModuleType(c, ts...))),
  subModules(parent, c, ts...) {}
  Vec<torch::Tensor> forward (Vec<torch::Tensor> tv) {
    tv = module(tv);
    tv = subModules(tv);
    return tv; }
  void reset() {}//override { this->reset(); }
  // torch::nn::Module* get(int idx) { if (idx == n) return (torch::nn::Module*)module; else return subModules.get(idx); }
};

template<template <int,int> class ModuleTemplate, typename ConfigType, int N> struct ModuleStack<ModuleTemplate, ConfigType, N, N> : torch::nn::Module {// torch::nn::Cloneable<ModuleStack<ModuleType, ConfigType, N, N>> {
  template<typename ...Ts> ModuleStack(Ts ...ts) {}
  void reset() {}//override {}
  Vec<torch::Tensor> forward (Vec<torch::Tensor> input) { return input; }
};

struct HookPointImpl : torch::nn::Module {// torch::nn::Cloneable<HookPointImpl> {
	void reset() {}//override { }
  HookPointImpl(int nnn, const std::string nm, const std::string pfx) : n(nnn), hook_name(nm), prefix(pfx) {}
  std::string name() { return hook_name; }
  std::string full_name() { return prefix + hook_name; }
  torch::Tensor forward(torch::Tensor x) {
    if (hook_added) {
      if (!act.numel()) { time = hi_res::now(); act = x; this->register_parameter(name(), act, false); }
      else act = x;
    }
    return x;
  }
  void reset_hooks() { }//act = torch::empty({ 0 }); }
  int n;
  std::string hook_name, prefix;
  torch::Tensor act;
  hi_res::time_point time {};
  bool hook_added { true };
};
TORCH_MODULE(HookPoint);

struct EmbedImpl : torch::nn::Module {
  EmbedImpl(GPTConfig cc) : cfg(cc) {}
  at::Tensor forward (at::Tensor x) { return W_E.index_select(0, x.flatten()).view({ x.size(0), x.size(1), C }); }// W_E.index({ x, torch::indexing::Slice() }); }
  void reset() {}//override {}
  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_E { this->register_parameter("W_E", torch::randn({ d_vocab, C })/std::sqrt(float(C))) };
};
TORCH_MODULE(Embed);

struct UnembedImpl : torch::nn::Module {
  UnembedImpl(GPTConfig cc) : cfg(cc) {}
  at::Tensor forward (at::Tensor x) { return torch::matmul(x, W_U); }
  void reset() {}//override {}
  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_U { this->register_parameter("W_U", torch::randn({ C, d_vocab - 1 })/std::sqrt(float(d_vocab))) };
};
TORCH_MODULE(Unembed);

struct PosEmbedImpl : torch::nn::Module {
  PosEmbedImpl(GPTConfig cc) : cfg(cc) {}
  at::Tensor forward (at::Tensor x) { return einsad::repeat(W_pos, "pos d_model -> batch pos d_model", { "batch", x.size(0) }); }
  void reset() {}//override {}
  GPTConfig cfg;
  int T { cfg["n_ctx"] }, C { cfg["d_model"] };
  torch::Tensor W_pos { this->register_parameter("W_pos", torch::randn({ T, C })/std::sqrt(float(C))) };
};
TORCH_MODULE(PosEmbed);

struct NandaAttentionImpl : torch::nn::Module {// torch::nn::Cloneable<NandaAttentionImpl> {
  NandaAttentionImpl(GPTConfig cc, int nnn, const std::string pfx) : cfg(cc), prefix(pfx + "attn."), n(nnn) {}
  torch::Tensor forward (torch::Tensor x) {
    // const int B { static_cast<int>(x.size(0)) };

    auto q = hook_q(torch::einsum("idh,bpd->bpih", { W_Q, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    auto k = hook_k(torch::einsum("idh,bpd->bpih", { W_K, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(torch::einsum("idh,bpd->bpih", { W_V, x }));// [nh C hs] @ [B T C] = [B T nh hs]

    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]

    auto attn_scores_pre = torch::matmul(q, k)/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    auto w = einsad::rearrange(W_O, "head_index d_head d_model -> d_model (head_index d_head)");
    auto out = torch::nn::functional::linear(z.reshape({ z.size(0), z.size(1), C }), w);
    return out;
  }

  void reset() {}//override {}
  
  GPTConfig cfg;
  std::string prefix;
  int n, nh { cfg["n_heads"] }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, hs { C/nh };
  torch::Tensor W_K { this->register_parameter("W_K", torch::randn({ nh, C, C/nh })/std::sqrt(float(C))) };
  torch::Tensor W_Q { this->register_parameter("W_Q", torch::randn({ nh, C, C/nh })/std::sqrt(float(C))) };
  torch::Tensor W_V { this->register_parameter("W_V", torch::randn({ nh, C, C/nh })/std::sqrt(float(C))) };
  torch::Tensor W_O { this->register_parameter("W_O", torch::randn({ nh, C/nh, C })/std::sqrt(float(C))) };
  HookPoint hook_k { this->register_module("hook_k", HookPoint(n, "hook_k", prefix)) };
  HookPoint hook_q { this->register_module("hook_q", HookPoint(n, "hook_q", prefix)) };
  HookPoint hook_v { this->register_module("hook_v", HookPoint(n, "hook_v", prefix)) };
  HookPoint hook_z { this->register_module("hook_z", HookPoint(n, "hook_z", prefix)) };
  HookPoint hook_pattern { this->register_module("hook_pattern", HookPoint(n, "hook_pattern", prefix)) };
  HookPoint hook_attn_scores { this->register_module("hook_attn_scores", HookPoint(n, "hook_attn_scores", prefix)) };
};
TORCH_MODULE(NandaAttention);

struct NandaMLPImpl : torch::nn::Module {// torch::nn::Cloneable<NandaMLPImpl> {
  NandaMLPImpl(GPTConfig cfg, int nnn, const std::string pfx) : c(cfg), prefix(pfx + "mlp."), n(nnn) {}
  void reset() {}//override {}
  torch::Tensor forward(torch::Tensor x) { 
    x = hook_pre(torch::matmul(x, W_in));
    x = reLU(x);
    x = hook_post(x);
    x = torch::matmul(x, W_out);
    return x;
  }
  GPTConfig c;
  std::string prefix;
  int n, C { c["d_model"] }, d_mlp { c["d_mlp"] };
  torch::Tensor W_in { this->register_parameter("W_in", torch::randn({ C, d_mlp })/std::sqrt(float(C))) };
  torch::nn::ReLU reLU { this->register_module("reLU", torch::nn::ReLU(torch::nn::ReLUOptions())) };
  torch::Tensor W_out { this->register_parameter("W_out", torch::randn({ d_mlp, C })/std::sqrt(float(C))) };
  HookPoint hook_pre { this->register_module("hook_pre", HookPoint(n, "hook_pre", prefix)) };
  HookPoint hook_post { this->register_module("hook_post", HookPoint(n, "hook_post", prefix)) };
};
TORCH_MODULE(NandaMLP);

template<int N, int m> struct NandaDecoderBlock : torch::nn::Module {
  NandaDecoderBlock() = default; 
  NandaDecoderBlock(GPTConfig cfg, int nnn, const std::string pfx) : c(cfg), n(nnn), prefix(pfx + "blocks." + std::to_string(m) + ".") {}
  Vec<torch::Tensor> forward (Vec<torch::Tensor> tv) {
    auto& x { tv[0] };
    x = hook_resid_mid(x + hook_attn_out(attn(hook_resid_pre(x))));
    x = hook_resid_post(x + hook_mlp_out(mlp(x)));
    return tv;
  }
  void reset() {}//override {}
  GPTConfig c;
  int n;
  std::string prefix;
  NandaAttention attn { this->register_module("attn", NandaAttention(c, n, prefix)) };
  NandaMLP mlp { this->register_module("mlp", NandaMLP(c, n, prefix)) };
  HookPoint hook_attn_out { this->register_module("hook_attn_out", HookPoint(n, "hook_attn_out", prefix)) };
  HookPoint hook_mlp_out { this->register_module("hook_mlp_out", HookPoint(n, "hook_mlp_out", prefix)) };
  HookPoint hook_resid_pre { this->register_module("hook_resid_pre", HookPoint(n, "hook_resid_pre", prefix)) };
  HookPoint hook_resid_mid { this->register_module("hook_resid_mid", HookPoint(n, "hook_resid_mid", prefix)) };
  HookPoint hook_resid_post { this->register_module("hook_resid_post", HookPoint(n, "hook_resid_post", prefix)) };
};

using NandaDecoderStackImpl = ModuleStack<NandaDecoderBlock, GPTConfig, 1>;
TORCH_MODULE(NandaDecoderStack);

template<int N = 0> GPTConfig prepareConfig(GPTConfig c, bool isTesting) { 
  if (isTesting) {
    print("changing params for testing");
    c["d_model"] = 4;
    c["d_mlp"] = 6;
    c["n_heads"] = 2;
  }
  return c;
}

struct TransformerImpl : torch::nn::Module {
  static constexpr const char* test_names[] { "test_loss", "train_loss", "train_acc", "test_acc", "excluded_loss_2D", "excluded_loss_2D_full", "excluded_acc_2D", "excluded_acc_2D_full", "excluded_loss_3D", "excluded_loss_3D_full", "excluded_acc_3D", "excluded_acc_3D_full", "trig_loss", "trig_loss_train", "trig_acc", "trig_acc_train", "sum_sq_weights", "cos_coeffs", "cos_sim", "fourier_embedding" };
  std::string name() { return "Transformer" + std::to_string(n); }
  TransformerImpl(GPTConfig c, c10::Device dev, int nnn = 0, Tester* tstr = nullptr) : cfg(prepareConfig(c, isTesting)), tester(tstr), n(nnn) { to(dev); }

  void importParameter(const std::string& name, const torch::Tensor& tensor) {
    auto actualName { name };
    size_t pos { 0UL };
    actualName = replaceAll(actualName, {"sAd"}, {"."});
    while (((pos = actualName.find("sAd")) != std::string::npos)) actualName.replace(pos++, 3, ".");

    for (auto& p : named_parameters()) {
      if (actualName == p.key() || p.key() == name) {
        auto t { tensor };
        if (isTesting) { for (int d {}; d < t.dim(); ++d) t = t.slice(d, 0, p.value().size(d)); }
        p.value().set_requires_grad(false);
        if (t.sizes() != p.value().sizes()) print("mismatch copying IValue", name, actualName, t.sizes(), p.value().sizes(), p.key());
        p.value().copy_(t.detach());
        break;
      }
    }
  }

  Vec<torch::Tensor> forward (Vec<torch::Tensor> tv) {
    // const int B { static_cast<int>(tv[0].size(0)) };
    auto embeddings { hook_embed(embed(tv[0])) }, positions { hook_pos_embed(pos_embed(tv[0])) };
    tv[0] = embeddings + positions;
    tv = blocks(tv);
    tv[0] = unembed(tv[0]);
    return tv;
  }

  std::pair<Vec<at::Tensor>, ActivationCache> run_with_cache(Vec<torch::Tensor> tv, Vec<std::string> h = {}) { 
    set_hooks(h); tv = forward(tv); 
    return { tv, getActivationCache() }; }

  void saveMetrics(TensorMap tMap) { 
    for (const auto& [key, t] : tMap) metrics["cp" + nDigits(n, 3) + "_" + key] = t.detach().to(torch::kCPU); }

  void preComputeMetrics(at::Tensor logits, TensorMap labels, ActivationCache& cache) {//c10::Device dev) {
    // logits = logits.slice(1, logits.size(1) - 1, logits.size(1)).squeeze();
    // auto cache { getActivationCache() };
    // if (mode == "train") saveMetric("excluded_loss", tester->getExcludedLoss(labels, unembed->W_U, blocks->module->mlp->W_out, cache));
    // saveMetrics(tester->getLoss(logits, *LABELS));
    // time_points.push_back(std::chrono::now());
    saveMetrics(tester->getRestrictedLoss(logits, labels, blocks->module->mlp->W_out, unembed->W_U, cache));
  }

  at::Tensor W_E() { return embed->W_E.slice(0, 0, -1); }
  at::Tensor W_neur() {
    constexpr auto mm { torch::matmul };
    return mm(mm(mm(W_E(), blocks->module->attn->W_V), blocks->module->attn->W_O), blocks->module->mlp->W_in);
  }
  at::Tensor W_logit() {
    constexpr auto mm { torch::matmul };
    return mm(blocks->module->mlp->W_out, unembed->W_U);
  }

  ActivationCache getActivationCache() {
    Vec<std::shared_ptr<HookPointImpl>> vec;
    for (const auto& item : named_modules())
      if (item.key().contains("hook_")) vec.push_back(std::dynamic_pointer_cast<HookPointImpl>(item.value()));
    std::sort(vec.begin(), vec.end(), [](auto l, auto r) { return l->time < r->time; });
    for (auto& hook : vec)
      for (auto& param : hook->named_parameters())
        cache.data.push_back({ hook->prefix + param.key(), param.value() });
    return cache;
  }

  void set_hooks(Vec<std::string> hook_names = {}) {
    for (const auto& item : named_modules()) {
      if (item.key().contains("hook")) {
        auto hook_point = std::dynamic_pointer_cast<HookPointImpl>(item.value());
        if (hook_names.size() && !hook_names.contains(hook_point->hook_name))
          hook_point->hook_added = false;
      }
    }
  }
  
  Vec<torch::Tensor> computeLoss(Vec<torch::Tensor> tv) {
    // print("tv[0] (loss) / tv.back() (trg)", tv[0].sizes(), tv.back().sizes());
    tv[0] = getCrossEntropy(tv[0].slice(1, 2, 3).view({ -1, tv[0].size(-1) }), tv.back().view({ -1 }));//      .squeeze().to(torch::kFloat64), tv.back());
    return tv;
  }

  double getFloatLoss(Vec<torch::Tensor> tv) { return tv[0].mean().item<double>(); }
  void reset() {}//override {}

  bool isTesting { false };
  GPTConfig cfg;
  AllData* allData;
  Tester* tester;
  int p { 113 }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, n { 0 };
  Embed embed { this->register_module("embed", Embed(cfg)) };
  PosEmbed pos_embed { this->register_module("pos_embed", PosEmbed(cfg)) };
  NandaDecoderStack blocks { this->register_module("blocks", NandaDecoderStack(cfg, n, "")) };
  Unembed unembed { this->register_module("unembed", Unembed(cfg)) };
  HookPoint hook_embed { this->register_module("hook_embed", HookPoint(n, "hook_embed", "")) };
  HookPoint hook_pos_embed { this->register_module("hook_pos_embed", HookPoint(n, "hook_pos_embed", "")) };
  TensorMap metrics;
  ActivationCache cache;
  // Vec<std::chrono::time_point> time_points { 20 };
};
TORCH_MODULE(Transformer);

#include <boost/python.hpp>

namespace bp = boost::python;
template <typename T> std::string fromStr(const T& x) { return { bp::extract<const char*>(x) }; }



template<int N> struct TransNBatcherImpl : torch::nn::Module {
  static constexpr int numModels { N };
  TransNBatcherImpl(Vec<std::shared_ptr<TransformerImpl>>& m, c10::Device dev) : models(m), device(dev) { //to(dev);
    // for (auto&  model : models) { //print("model name and tok emb weights", model->name(), model->embed->weight[0][0].item<float>(), model->embed->weight[1][1].item<float>()); 
    // this->register_module(model->name(), model); }
  }
  Vec<Vec<torch::Tensor>> forward(Vec<torch::Tensor> tv) {
    Vec<Vec<torch::Tensor>> logits;
    for (auto& model: models) logits.push_back(model->forward(tv));
    return logits;
  }
  std::pair<Vec<Vec<at::Tensor>>, Vec<ActivationCache>> run_with_cache(Vec<at::Tensor> tv, Vec<std::string> h = {}) {
    auto defaultStream = at::cuda::getDefaultCUDAStream();
    Vec<at::cuda::CUDAStream> streams;
    Vec<Vec<at::Tensor>> tvs { models.size() };
    Vec<ActivationCache> caches { models.size() };

    // std::pair<Vec<Vec<at::Tensor>>, Vec<ActivationCache>> ret;
    // auto& [logits, activations] = ret;
    for (int i {}; i < models.size(); ++i) {
      streams.push_back(at::cuda::getStreamFromPool());
      {
      torch::NoGradGuard guard;
      at::cuda::setCurrentCUDAStream(streams[i]);
      Vec<at::Tensor> tvNew { { tv[0].clone().to(device, torch::kLong, true) } };
      auto [logit,activation] = models[i]->run_with_cache(tvNew, h);
      tvs[i] = logit;
      caches[i] = activation;
      }
    }
    torch::cuda::synchronize();
    return { tvs, caches };
    // for (auto& model: models) {
    //   auto [logit,activation] = model->run_with_cache(tv, h);
    //   logits.push_back(logit);
    //   activations.push_back(activation);
    // }
    // return ret;
  }
  Vec<Vec<torch::Tensor>> computeLoss(Vec<Vec<torch::Tensor>> tv, torch::Tensor label) {
    for (int i {}; i < tv.size(); ++i) {
      tv[i].push_back(label);
      tv[i] = models[i]->computeLoss(tv[i]);
    }
    return tv;
  }
  void preComputeMetrics(Vec<Vec<at::Tensor>> logits, TensorMap labels, Vec<ActivationCache> caches) {
    torch::NoGradGuard guard;
    auto logit = logits.begin();
    auto cache = caches.begin();
    for (auto model = models.begin(); model != models.end(); ++model, ++logit, ++cache) 
      (*model)->preComputeMetrics((*logit)[0], labels, (*cache));
  }

  void save(TensorMap& allMetrics) {
    for (const auto& model : models) allMetrics.insert(model->metrics.begin(), model->metrics.end()); }
  Vec<std::shared_ptr<TransformerImpl>>& models;
  c10::Device device;
};

using TransBatcherImpl = TransNBatcherImpl<20>;
TORCH_MODULE(TransBatcher);

struct CheckpointProcessor {
  static constexpr int p { 113 };
  static constexpr const char* tests[] { "embed", "pos_embed" };
  static constexpr const size_t n_tests { sizeof(tests)/sizeof(const char*) };
  using ModelList = Vec<Vec<std::pair<std::string, at::Tensor>>>;
  static inline const std::string name() { return "CheckpointX"; }
  CheckpointProcessor();
  CheckpointProcessor(bp::dict);
  CheckpointProcessor(bp::str);
  CheckpointProcessor(bp::list);
  void load_checkpoints(bp::list state_dicts);
  void load_checkpoints(torch::jit::script::Module state_dicts);
  template<class M> void load_model(int n, M& model) { for (auto& [name,ten] : dicts[n]) model->importParameter(name, ten); }
  void load_model(int);
  int getCheckpointNumber(const std::string& full_name);
  void run_tests();
  void load_models();
  void load_config(bp::dict config);
  void doForward(TransBatcher& model, int n);
  void compare_cache(bp::dict);

  GPTConfig cfg {};
  c10::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
  AllData allData { makeAllData(device) };
  Tester tester {};
  TensorMap& EXAMPLES { allData.dataset };
  TensorMap& LABELS { allData.labels };
  ModelList dicts;
  std::shared_ptr<TransformerImpl> cpModel { nullptr };
  std::string nam { "cheeze" };
  double getNumber() { return number; }
  void setNumber(double x) { number = x; }
  double number = 9.0;
};

static boost::shared_ptr<CheckpointProcessor> makeCPProcessor(const bp::object& data) {
  boost::shared_ptr<CheckpointProcessor> obj;
  if (PyDict_Check(data.ptr())) {
    obj.reset(new CheckpointProcessor(bp::dict(data)));
    return obj;
  }
  else if (PyList_Check(data.ptr())) {
    obj.reset(new CheckpointProcessor(bp::list(data)));
    return obj;
  }
  else if (bp::extract<const char*>(data).check()) {
    obj.reset(new CheckpointProcessor(bp::str(data)));
    return obj;
  }
  else return nullptr;
}