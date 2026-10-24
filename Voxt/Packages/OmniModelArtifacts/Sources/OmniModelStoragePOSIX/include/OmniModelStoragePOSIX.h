#ifndef OMNI_MODEL_STORAGE_POSIX_H
#define OMNI_MODEL_STORAGE_POSIX_H

int omni_model_storage_flock(int descriptor, int operation);
int omni_model_storage_exchange(const char *staging, const char *destination);

#endif
