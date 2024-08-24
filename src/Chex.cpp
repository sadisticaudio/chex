#include <einops.hpp>
#include "Chex.h"
#include <torch/script.h>
#include "/usr/local/cuda-11.8/targets/x86_64-linux/include/cuda_runtime_api.h"

namespace einsad {
  template<class... A> at::Tensor rearrange(at::Tensor x, std::string p, A... a) { return einops::rearrange(x, p, a...); }
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a) { return einops::repeat(x, p, a); }
  template<class... Ts> at::Tensor einsum(std::string pattern, Ts... ts) { return einops::einsum(pattern, ts...); }
}

using ModelList = typename CheckpointProcessor::ModelList;

template <typename T> std::string getPythonClass(const T& x) { return fromStr(x.attr("__class__").attr("__name__")); }
#include<unistd.h>
CheckpointProcessor::CheckpointProcessor(){ print("WHY ARE WE DEFAULT CONSTRUCTING CPPROCESSOR??"); }
CheckpointProcessor::CheckpointProcessor(bp::dict full_run_data) {
  load_config(bp::dict(full_run_data["config"].attr("to_dict")()));
  load_checkpoints(bp::list(full_run_data["checkpoints"]));// list of OrderedDict of [str, Tensor]
  print("in CP Constructor");

  load_model(dicts.size() - 1);

  run_tests();
}

CheckpointProcessor::CheckpointProcessor(bp::str pyPath) {
  std::string path { fromStr(pyPath) };
  print("IN CP(bp::str) CTOR ... bp::str path", path);
  torch::jit::script::Module tensors = torch::jit::load(path);
  print("jit tensors loaded, num", tensors.named_parameters().size());

  load_checkpoints(tensors);// list of OrderedDict of [str, Tensor]
  run_tests();
}

CheckpointProcessor::CheckpointProcessor(bp::list pyList) { load_checkpoints(pyList); run_tests(); }

void CheckpointProcessor::run_tests() {

  // NOT SURE WHERE TO PUT THIS
  print("pushing EXAMPLES & LABELS to device", device);
  for (auto& ex : EXAMPLES) ex.second = ex.second.to(device);
  for (auto& lbl : LABELS) lbl.second = lbl.second.to(device);

  int modelNum = dicts.size() - 1;
  auto model { std::make_shared<TransformerImpl>(cfg, device, modelNum, &tester) };
  load_model(modelNum, model);

  cpModel = model;
  load_models();
}

void CheckpointProcessor::load_checkpoints(bp::list state_dicts) {
  
  for (auto sd { bp::stl_input_iterator<bp::dict>(state_dicts) }; sd != bp::stl_input_iterator<bp::dict>(); ++sd) {
    Vec<std::pair<std::string, at::Tensor>> dict;
    auto name = bp::stl_input_iterator<bp::str>(sd->keys());
    auto tensor = bp::stl_input_iterator<bp::api::object>(sd->values());

    for (; name != bp::stl_input_iterator<bp::str>(); ++name, ++tensor) {
      dict.push_back({ fromStr(*name), THPVariable_Unpack((*tensor).ptr()) });
    }
    dicts.push_back(dict);
  }
}

void CheckpointProcessor::compare_cache(bp::dict pyCache) {
  auto input = torch::tensor({ { 60L, 90L, 113L } }).to(device), label = torch::tensor({ { 37L } }).to(device);
  print("testing input", input, "label", label);
  auto logits = cpModel->forward(Vec<at::Tensor>({ input }));
  logits.push_back(label);
  auto losses = cpModel->computeLoss(logits);
  print("loss", losses[0].mean().item<double>());

  auto cache = cpModel->getActivationCache();

  auto items = pyCache.attr("items")();
  auto pyPair = bp::stl_input_iterator<bp::tuple>(items);
  for (; pyPair != bp::stl_input_iterator<bp::tuple>(); ++pyPair) {
    bp::tuple kv = *pyPair;
    auto name = fromStr(bp::str(kv[0]));
    auto t = THPVariable_Unpack((bp::api::object(kv[1])).ptr());
    if (cache.contains(name)) {
      if (cache[name].sizes() != t.sizes()) print("WHOAAAA!!!!, MISMATCH..", name, cache[name].sizes(), t.sizes());
      else {
        if(torch::allclose(t.to(torch::kCPU), cache[name].to(torch::kCPU), 1E-03F, 1E-05F)) print(name, " caches are the same");
        else print(name + ": caches are not equal, python", get_first_elements(t, 4), "c++", get_first_elements(cache[name], 4));
      }
    }
  }
}

int CheckpointProcessor::getCheckpointNumber(const std::string& full_name) {
  size_t pos {}, endPos {};
  while(pos < full_name.size() && !std::isdigit(full_name[pos])) pos++;
  endPos = pos + 1UL;
  while(endPos < full_name.size() && std::isdigit(full_name[endPos])) endPos++;
  return std::stoi(full_name.substr(pos, endPos));
}

void CheckpointProcessor::load_checkpoints(torch::jit::script::Module dict_model1) {
  auto dict_model = dict_model1.clone();
  auto all_dicts = dict_model.named_parameters();
  auto itr = all_dicts.begin();
  int chNum { getCheckpointNumber((*itr).name) };

  for (; itr != all_dicts.end();) {
    Vec<std::pair<std::string, at::Tensor>> dict;
    for(;itr != all_dicts.end() && chNum == getCheckpointNumber((*itr).name); ++itr) {
      dict.push_back({ (*itr).name, (*itr).value });
    }
    if (dict.size() > 0) dicts.push_back(dict);
    else print("was about to push_back a checkpoint with dict.size() == 0");
    if (itr != all_dicts.end() && chNum != getCheckpointNumber((*itr).name)) chNum = getCheckpointNumber((*itr).name);
  }
}

void CheckpointProcessor::doForward(TransBatcher& batcher, int n) {
  auto& b { *batcher };
  hi_res::time_point begin = hi_res::now();
  auto record_time = [&](auto name) {
    auto end = hi_res::now();
    std::cout << "Time difference for " << name << " = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    // std::cout << "Time difference for " << name << " = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    begin = end;
  };
  auto [logits, cache] = b.run_with_cache({{ EXAMPLES["all"] }}, {{ "hook_post", "hook_resid_mid" }});
  record_time("\nrun_with_cache");
  b.preComputeMetrics(logits, LABELS, cache);
  record_time("precomputeMetrics");
}

void CheckpointProcessor::load_config(bp::dict config) {
  cfg["lr"] = 1e-3f;
  cfg["frac_train"] = 0.3f;
  constexpr int numTrain { static_cast<int>(p * p * 0.3f) }, numTest { p * p - numTrain };
  forEach([&](auto name){ cfg[name] = bp::extract<int>(config[name]); }, 
    "d_model", "d_vocab", "d_vocab_out", "n_ctx", "d_head", "d_mlp", "n_heads", "n_layers");
}

void CheckpointProcessor::load_model(int n) {
  cpModel = std::make_shared<TransformerImpl>(cfg, device, n, &tester);
  if (cpModel) load_model(n, cpModel);
}

void CheckpointProcessor::load_models() {
  std::map<std::string, torch::Tensor> allMetrics;
  for (int n {}; n < dicts.size(); n += TransBatcherImpl::numModels) {
    Vec<std::shared_ptr<TransformerImpl>> models;
    for (int i {}; n + i < dicts.size() && i < TransBatcherImpl::numModels; ++i) {
      auto& model = models.push_back(std::make_shared<TransformerImpl>(cfg, device, n + i, &tester));
      load_model(n + i, model);
    }
    TransBatcher batcher { models, device };
    batcher->eval();
    doForward(batcher, n);
    batcher->save(allMetrics);
  }

  std::vector<std::pair<std::string, torch::Tensor>> vec { allMetrics.begin(), allMetrics.end() };
  std::set<std::string> metricSet;
  for (const auto& [name, t] : vec) metricSet.insert(name.substr(name.find('_') + 1UL));
  std::vector<std::string> metrics { metricSet.begin(), metricSet.end() };
  std::sort(metrics.begin(), metrics.end(), [](auto l, auto r) { auto lsub = l.substr(l.find('_')); auto rsub = r.substr(r.find('_')); return lsub == rsub ? l > r : lsub < rsub; });
  for (const auto& metric : metrics) {
    print("\n\n" + metric + "\n");
    for (const auto& [name, t] : vec) if (name.substr(name.find('_') + 1UL) == metric) std::cout << t.item<float>() << " ";
  }
  saveContainer(vec, "../saved/metrics");
}

