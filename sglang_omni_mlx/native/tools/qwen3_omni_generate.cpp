// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/serving.h"
#include <iostream>

int main(int count, char **values) {
  using namespace qwen3_omni;
  try {
    auto arguments = parse_arguments(count, values, false);
    if (arguments.version) {
      std::cout << version_json().dump() << '\n';
      return 0;
    } else {
    }
    std::vector<ChatRequest> requests;
    for (const auto &path : arguments.requests) {
      if (std::filesystem::file_size(path) > 64 * 1024 * 1024) {
        throw std::invalid_argument("request exceeds 64 MiB");
      } else {
        requests.push_back(parse_request(omni_common::ReadJson(path),
                                         arguments.model_name,
                                         arguments.allowed_paths));
      }
    }
    mx::set_default_device(mx::Device::gpu);
    OmniModel model(arguments.model_path);
    for (const auto &request : requests) {
      auto completion = make_completion(request.model);
      model.generate(
          request,
          [&](const GenerationChunk &chunk) { completion.append(chunk); }, {});
      std::cout << completion.json().dump() << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
