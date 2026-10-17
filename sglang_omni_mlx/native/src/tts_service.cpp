// SPDX-License-Identifier: Apache-2.0
#include "tts_service.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <typeinfo>

#include "http.h"

namespace ming_tts {

namespace mx = mlx::core;

namespace {

constexpr int kCfmSteps = 10;
constexpr int kInitialChunkPatches = 2;
constexpr int kSteadyChunkPatches = 4;
constexpr size_t kMaxBodyBytes = 1024 * 1024;

std::string Pcm(const mx::array &waveform) {
  if (waveform.size() == 0) {
    return {};
  } else {
  }
  const auto samples =
      mx::astype(mx::clip(mx::astype(waveform, mx::float32) * 32768.0f,
                          mx::array(-32768.0f), mx::array(32767.0f)),
                 mx::int16);
  mx::eval(samples);
  return {reinterpret_cast<const char *>(samples.data<int16_t>()),
          samples.size() * sizeof(int16_t)};
}

std::string WavHeader(uint32_t audio_bytes, uint32_t sample_rate) {
  std::string header;
  const auto append = [&](uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      header.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
    }
  };
  header += "RIFF";
  append(36 + audio_bytes, 4);
  header += "WAVEfmt ";
  append(16, 4);
  append(1, 2);
  append(1, 2);
  append(sample_rate, 4);
  append(sample_rate * 2, 4);
  append(2, 2);
  append(16, 2);
  header += "data";
  append(audio_bytes, 4);
  return header;
}

} // namespace

SpeechRequest SpeechRequest::Parse(const nlohmann::json &request) {
  static const std::set<std::string> fields{
      "model",  "input",        "response_format", "stream",
      "voice",  "speed",        "max_new_tokens",  "temperature",
      "prompt", "instructions", "stage_params"};
  if (!request.is_object()) {
    throw std::invalid_argument("request must be a JSON object");
  } else {
  }
  for (const auto &[name, value] : request.items()) {
    if (!fields.contains(name)) {
      throw std::invalid_argument("unsupported speech field: " + name);
    } else {
    }
  }
  request.at("model").get<std::string>();
  const auto text = request.at("input").get<std::string>();
  const auto value_or_default = [](const nlohmann::json &source,
                                   const std::string &name,
                                   const nlohmann::json &default_value) {
    const auto value = source.value(name, nlohmann::json());
    return value.is_null() ? default_value : value;
  };
  const auto prompt =
      value_or_default(request, "prompt", "").get<std::string>();
  const auto instructions =
      value_or_default(request, "instructions", "").get<std::string>();
  const auto format = request.value("response_format", std::string("wav"));
  const bool stream = request.value("stream", false);
  const auto length = value_or_default(request, "max_new_tokens", 200);
  if ((format != "wav" && format != "pcm") || (stream && format != "pcm")) {
    throw std::invalid_argument("use wav or pcm; streaming requires pcm");
  } else if (request.value("voice", std::string("default")) != "default" ||
             request.value("speed", 1.0) != 1.0) {
    throw std::invalid_argument("only voice=default and speed=1 are supported");
  } else if (!length.is_number_integer() || length < 1 || length > 256) {
    throw std::invalid_argument("max_new_tokens must be an integer in [1,256]");
  } else {
  }
  const auto stages =
      value_or_default(request, "stage_params", nlohmann::json::object());
  if (!stages.is_object() || stages.size() > stages.count("tts_engine")) {
    throw std::invalid_argument(
        "only tts_engine sampling parameters are supported");
  } else {
  }
  const auto sampling = stages.value("tts_engine", nlohmann::json::object());
  if (!sampling.is_object()) {
    throw std::invalid_argument("tts_engine must be an object");
  } else {
  }
  for (const auto &[name, value] : sampling.items()) {
    if (name != "cfg" && name != "sigma" && name != "temperature") {
      throw std::invalid_argument("unsupported sampling field: " + name);
    } else {
    }
  }
  const auto temperature_value =
      sampling.value("temperature", nlohmann::json());
  const float temperature =
      temperature_value.is_null()
          ? value_or_default(request, "temperature", 0.0f).get<float>()
          : temperature_value.get<float>();
  const float guidance_scale =
      value_or_default(sampling, "cfg", 2.0f).get<float>();
  const float sigma = value_or_default(sampling, "sigma", 0.25f).get<float>();
  if (!std::isfinite(guidance_scale) || guidance_scale < 1e-5f ||
      guidance_scale == 1.0f) {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only guided-branch cfg values; "
        "cfg must be >= 1e-5 and not equal to 1.0");
  } else if (!std::isfinite(sigma) || sigma < 0 ||
             !std::isfinite(temperature) || temperature < 0) {
    throw std::invalid_argument(
        "sigma and temperature must be finite and nonnegative");
  } else {
    return {text,           prompt,          instructions,
            stream,         format == "wav", length.get<int>(),
            guidance_scale, sigma,           temperature};
  }
}

MingSpeechService::MingSpeechService(const std::filesystem::path &directory) {
  std::promise<void> loaded;
  auto ready = loaded.get_future();
  thread_ =
      std::thread(&MingSpeechService::Run, this, directory, std::move(loaded));
  try {
    ready.get();
  } catch (...) {
    thread_.join();
    throw;
  }
}

MingSpeechService::~MingSpeechService() {
  CancelAll();
  thread_.join();
}

void MingSpeechService::Register(mg_context *context, const std::string &) {
  mg_set_request_handler(
      context, "/v1/audio/speech$",
      [](mg_connection *connection, void *service) {
        return static_cast<MingSpeechService *>(service)->HandleSpeech(
            connection);
      },
      this);
}

int MingSpeechService::HandleSpeech(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405, {{"detail", "use POST"}});
  } else {
  }
  try {
    std::string body;
    char buffer[4096];
    int count;
    while ((count = mg_read(connection, buffer, sizeof(buffer))) > 0) {
      body.append(buffer, static_cast<size_t>(count));
      if (body.size() > kMaxBodyBytes) {
        throw std::invalid_argument("request exceeds 1 MiB");
      } else {
      }
    }
    auto job = std::make_shared<Job>();
    job->connection = connection;
    job->request = SpeechRequest::Parse(nlohmann::json::parse(body));
    auto completion = job->completion.get_future();
    int rejected_status = 0;
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || job_) {
        rejected_status = stopping_ ? 503 : 429;
      } else {
        job_ = job;
      }
    }
    if (rejected_status) {
      omni_server::WriteResponse(
          connection, rejected_status,
          rejected_status == 503 ? "Service Unavailable" : "Too Many Requests",
          "application/json", "{\"detail\":\"server unavailable or busy\"}");
      return rejected_status;
    } else {
      wake_.notify_one();
      return completion.get();
    }
  } catch (const nlohmann::json::exception &error) {
    return omni_server::BadRequest(connection, error.what());
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  }
}

std::map<std::string, int> MingSpeechService::RequestStates() const {
  std::lock_guard lock(mutex_);
  if (job_) {
    return {{"running", 1}};
  } else {
    return {};
  }
}

void MingSpeechService::CancelAll() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_one();
}

void MingSpeechService::Run(const std::filesystem::path &directory,
                            std::promise<void> loaded) {
  std::unique_ptr<MingFrontend> frontend;
  std::unique_ptr<MingGenerator> generator;
  std::unique_ptr<MingAudioDecoder> decoder;
  try {
    frontend = std::make_unique<MingFrontend>(directory);
    generator = std::make_unique<MingGenerator>(directory);
    decoder = std::make_unique<MingAudioDecoder>(directory);
    mx::synchronize();
    mx::clear_cache();
    loaded.set_value();
  } catch (...) {
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || job_; });
      if (!job_) {
        return;
      } else {
        job = job_;
      }
    }
    int status = 500;
    std::string detail;
    try {
      status = Generate(*job, *frontend, *generator, *decoder);
    } catch (const std::invalid_argument &error) {
      std::cerr << "Speech generation failed: " << typeid(error).name() << '\n';
      status = 400;
      detail = error.what();
    } catch (const std::exception &error) {
      std::cerr << "Speech generation failed: " << typeid(error).name() << '\n';
      detail = error.what();
    }
    try {
      mx::synchronize();
      mx::clear_cache();
    } catch (const std::exception &error) {
      std::cerr << "Speech cleanup failed: " << typeid(error).name() << '\n';
      status = 500;
      detail = error.what();
    }
    {
      std::lock_guard lock(mutex_);
      job_.reset();
    }
    if (status != 200) {
      if (!job->headers_sent) {
        omni_server::WriteJson(job->connection, status, {{"detail", detail}});
      } else {
      }
    } else if (job->request.stream) {
      mg_send_chunk(job->connection, "", 0);
    } else {
      mg_printf(job->connection,
                "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                "Connection: close\r\nX-Sample-Rate: %d\r\nX-Channels: 1\r\n"
                "X-Bit-Depth: 16\r\n\r\n",
                job->request.wav ? "audio/wav" : "audio/pcm", job->audio.size(),
                job->sample_rate);
      mg_write(job->connection, job->audio.data(), job->audio.size());
    }
    // Note (altale): Return the sent HTTP status, but leave failed streams
    // unterminated.
    job->completion.set_value(job->headers_sent ? 200 : status);
  }
}

int MingSpeechService::Generate(Job &job, const MingFrontend &frontend,
                                const MingGenerator &generator,
                                const MingAudioDecoder &decoder) {
  if (stopping_) {
    throw std::runtime_error("server is stopping");
  } else {
  }
  const auto &request = job.request;
  const auto input_ids =
      frontend.BuildPrompt(request.text, request.prompt, request.instructions);
  auto state = generator.Start(input_ids, request.max_new_tokens);
  const auto timesteps = CfmTimesteps(kCfmSteps);
  const auto &acoustic = generator.acoustic();
  AudioDecoderState audio_state;
  std::vector<mx::array> pending;
  int chunk_patches = kInitialChunkPatches;
  while (!state.finished && !stopping_) {
    const auto noise =
        mx::random::normal({1, acoustic.latent_dim(), acoustic.patch_size()});
    const auto sde_random =
        mx::random::normal({timesteps.shape(0) - 2, 1, acoustic.patch_size(),
                            acoustic.latent_dim()});
    pending.push_back(generator.Step(state, noise, timesteps, sde_random,
                                     request.guidance_scale, request.sigma,
                                     request.temperature));
    if (request.stream &&
        (static_cast<int>(pending.size()) >= chunk_patches || state.finished)) {
      const auto waveform =
          decoder.DecodeChunk(mx::expand_dims(mx::concatenate(pending, 0), 0),
                              audio_state, state.finished);
      pending.clear();
      chunk_patches = kSteadyChunkPatches;
      const auto pcm = Pcm(waveform);
      if (pcm.empty()) {
        continue;
      } else if (!job.headers_sent) {
        job.headers_sent = true;
        if (mg_printf(
                job.connection,
                "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                "Transfer-Encoding: chunked\r\nConnection: close\r\n"
                "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n\r\n",
                decoder.sample_rate()) <= 0) {
          return 500;
        } else {
        }
      } else {
      }
      if (mg_send_chunk(job.connection, pcm.data(),
                        static_cast<unsigned int>(pcm.size())) < 0) {
        return 500;
      } else {
      }
    } else {
    }
  }
  if (stopping_) {
    throw std::runtime_error("server is stopping");
  } else if (!request.stream) {
    auto audio = Pcm(
        decoder.DecodeFull(mx::expand_dims(mx::concatenate(pending, 0), 0)));
    if (request.wav) {
      audio = WavHeader(static_cast<uint32_t>(audio.size()),
                        decoder.sample_rate()) +
              audio;
    } else {
    }
    job.audio = std::move(audio);
    job.sample_rate = decoder.sample_rate();
  } else {
  }
  return 200;
}

} // namespace ming_tts
