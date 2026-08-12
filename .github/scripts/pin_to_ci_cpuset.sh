# Pin the sourcing shell (and children) to the runner lane's OMNI_CI_CPUSET;
# no-op when unset (forks, local runs).
# note (Jiaxin Deng): docker container options cannot read the runner-local
# .env value, so the reachable layer is the script shell, not --cpuset-cpus.
if [ -n "${OMNI_CI_CPUSET:-}" ]; then
  if ! taskset -pc "${OMNI_CI_CPUSET}" $$ > /dev/null; then
    echo "::error::failed to pin to OMNI_CI_CPUSET='${OMNI_CI_CPUSET}'"
    exit 2
  fi
  # note (Jiaxin Deng): readback catches the kernel silently narrowing the
  # mask (e.g. offline CPUs in the lane spec), mirroring the perf fixture.
  _pin_effective="$(taskset -pc $$ | awk '{print $NF}')"
  if [ "${_pin_effective}" != "${OMNI_CI_CPUSET}" ]; then
    echo "::error::pinned mask '${_pin_effective}' != requested OMNI_CI_CPUSET='${OMNI_CI_CPUSET}'"
    exit 2
  fi
  unset _pin_effective
fi
