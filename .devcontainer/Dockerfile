FROM frankleeeee/sglang-omni:dev AS base

ENV PYTHONUNBUFFERED=1

# Default to non 1000 uid to avoid default 1000 uid collsion
ARG HOST_UID=1003
ARG HOST_GID=1003
# fallback uid to swap if HOST_UID is taken
ARG DEV_UID_FALLBACK=1100
ARG DEV_GID_FALLBACK=1100

RUN set -eux; \
    conflict_group="$(getent group "${HOST_GID}" | cut -d: -f1 || true)"; \
    if [ -n "${conflict_group}" ] && [ "${conflict_group}" != "devuser" ]; then \
        groupmod -g "${DEV_GID_FALLBACK}" "${conflict_group}"; \
    fi; \
    if ! getent group devuser >/dev/null; then \
        groupadd -g "${HOST_GID}" devuser; \
    else \
        groupmod -g "${HOST_GID}" devuser; \
    fi; \
    conflict_user="$(getent passwd "${HOST_UID}" | cut -d: -f1 || true)"; \
    if [ -n "${conflict_user}" ] && [ "${conflict_user}" != "devuser" ]; then \
        old_group="$(id -gn "${conflict_user}")"; \
        usermod -u "${DEV_UID_FALLBACK}" -g "${DEV_GID_FALLBACK}" "${conflict_user}"; \
        if [ -d "/home/${conflict_user}" ]; then \
            chown -R "${DEV_UID_FALLBACK}:${DEV_GID_FALLBACK}" "/home/${conflict_user}"; \
        fi; \
    fi; \
    if id -u devuser >/dev/null 2>&1; then \
        usermod -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/zsh devuser; \
    else \
        useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/zsh devuser; \
    fi; \
    if [ -d /home/devuser ]; then \
        chown -R "${HOST_UID}:${HOST_GID}" /home/devuser; \
    fi

RUN apt-get update \
    && apt-get install -y sudo \
    && echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# copy host shell configs
RUN if [ -d /root/.oh-my-zsh ]; then cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh; fi \
    && if [ -f /root/.zshrc ]; then cp /root/.zshrc /home/devuser/.zshrc; fi \
    && if [ -f /root/.vimrc ]; then cp /root/.vimrc /home/devuser/.vimrc; fi \
    && if [ -f /root/.tmux.conf ]; then cp /root/.tmux.conf /home/devuser/.tmux.conf; fi \
    && if [ -f /home/devuser/.zshrc ]; then sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc; fi \
    && chown -R devuser:devuser /home/devuser

WORKDIR /sgl-workspace/sglang-omni
RUN install -d -o devuser -g devuser /sgl-workspace

USER devuser

# install uv
RUN if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi

ENV PATH="/home/devuser/.local/bin:${PATH}"

FROM base AS dev
