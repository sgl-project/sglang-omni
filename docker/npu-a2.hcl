variable "SGLANG_TAG" {}
variable "CANN_VERSION" {}
variable "CANN_IMAGE" {}
variable "SGLANG_IMAGE" {}
variable "MODELSCOPE_VERSION" {}
variable "EVALSCOPE_VERSION" {}

target "sglang-910b" {
  context    = ".cache/sglang-910b"
  dockerfile = "npu.Dockerfile"
  platforms  = ["linux/arm64"]
  contexts = {
    (split("@", CANN_IMAGE)[0]) = "docker-image://${CANN_IMAGE}"
  }
  args = {
    DEVICE_TYPE        = "910b"
    CANN_VERSION       = CANN_VERSION
    SGLANG_TAG         = SGLANG_TAG
    MODELSCOPE_VERSION = MODELSCOPE_VERSION
    EVALSCOPE_VERSION  = EVALSCOPE_VERSION
  }
}

target "omni-910b" {
  context    = "."
  dockerfile = "docker/npu.Dockerfile"
  platforms  = ["linux/arm64"]
  contexts   = { (SGLANG_IMAGE) = "target:sglang-910b" }
  args       = { SGLANG_IMAGE = SGLANG_IMAGE }
}
