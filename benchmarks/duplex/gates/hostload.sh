#!/bin/bash
# hostload.sh: host CPU sampler, every 0.25 s until killed. Columns: /proc/uptime seconds, the four /proc/loadavg fields,
# the aggregate /proc/stat cpu counters, procs_running. When more than DUMP_RUNNABLE threads are runnable it also prints a
# DUMP line with the commands that own them. gate.sh starts it next to each ladder run when HOSTLOAD=1.
DUMP_RUNNABLE=${DUMP_RUNNABLE:-300}
while true; do
  running=$(cut -d' ' -f4 /proc/loadavg | cut -d/ -f1)
  echo "$(cut -d' ' -f1 /proc/uptime) $(cut -d' ' -f1-4 /proc/loadavg) $(head -1 /proc/stat | cut -d' ' -f2-) procs_running $running"
  if [ "$running" -gt "$DUMP_RUNNABLE" ]; then
    echo "DUMP $(cut -d' ' -f1 /proc/uptime) $(ps -eLo stat=,args= | awk '$1 ~ /^R/ {$1=""; print substr($0,1,120)}' | sort | uniq -c | sort -rn | head -8 | tr '\n' '|')"
  fi
  sleep 0.25
done
