# SPDX-License-Identifier: Apache-2.0 AND MIT
# Portions derived from https://github.com/step-out/C3 (Copyright (c) 2025-2026 C3 Benchmark Authors, MIT License; see benchmarks/c3/LICENSE).
"""Descriptor-relative O_NOFOLLOW filesystem containment and atomic writers for C3."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
from pathlib import Path, PurePosixPath

from pydantic import JsonValue


def split_root_components_from_filesystem_anchor(
    raw_root_path: Path,
) -> tuple[str, tuple[str, ...]]:
    """Expand user home, reject parent traversal, and return anchor plus components."""
    raw_text = str(raw_root_path)
    if "\x00" in raw_text or "\\" in raw_text:
        raise ValueError(f"Unsafe root directory path: {raw_root_path!r}")
    else:
        pass
    expanded_path = raw_root_path.expanduser()
    absolute_candidate = (
        expanded_path if expanded_path.is_absolute() else (Path.cwd() / expanded_path)
    )
    filesystem_anchor = absolute_candidate.anchor or "/"
    directory_components: list[str] = []
    for path_segment in absolute_candidate.parts[1:]:
        if path_segment in ("", "."):
            continue
        elif path_segment == "..":
            raise ValueError(
                f"Parent traversal '..' is not permitted in root path: {raw_root_path}"
            )
        else:
            directory_components.append(path_segment)
    return filesystem_anchor, tuple(directory_components)


def open_root_directory_nofollow(
    raw_root_path: Path, *, create_if_missing: bool = False
) -> tuple[int, Path]:
    """Acquire a directory descriptor from '/' step-by-step with O_NOFOLLOW and O_DIRECTORY."""
    filesystem_anchor, directory_components = (
        split_root_components_from_filesystem_anchor(raw_root_path)
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_directory_descriptor = os.open(filesystem_anchor, directory_flags)
    traversed_components: list[str] = []
    try:
        for directory_component in directory_components:
            traversed_components.append(directory_component)
            try:
                next_directory_descriptor = os.open(
                    directory_component,
                    directory_flags,
                    dir_fd=current_directory_descriptor,
                )
            except FileNotFoundError:
                if not create_if_missing:
                    raise
                else:
                    try:
                        os.mkdir(
                            directory_component,
                            0o755,
                            dir_fd=current_directory_descriptor,
                        )
                    except FileExistsError:
                        pass
                    next_directory_descriptor = os.open(
                        directory_component,
                        directory_flags,
                        dir_fd=current_directory_descriptor,
                    )
            os.close(current_directory_descriptor)
            current_directory_descriptor = next_directory_descriptor
    except OSError as os_error:
        os.close(current_directory_descriptor)
        if os_error.errno in (errno.ELOOP, errno.ENOTDIR):
            traversed_display = Path(filesystem_anchor, *traversed_components)
            raise ValueError(
                f"Symlink or non-directory component {traversed_display} is not permitted in root path: {raw_root_path}"
            ) from None
        else:
            raise
    except BaseException:
        os.close(current_directory_descriptor)
        raise

    verified_path = Path(filesystem_anchor, *directory_components)
    return current_directory_descriptor, verified_path


def validate_non_symlink_root_directory(
    raw_root_path: Path, *, create_if_missing: bool = False
) -> Path:
    """Validate every component from '/' using O_NOFOLLOW descriptors without calling resolve()."""
    if create_if_missing:
        directory_descriptor, verified_path = open_root_directory_nofollow(
            raw_root_path, create_if_missing=True
        )
        os.close(directory_descriptor)
        return verified_path
    else:
        filesystem_anchor, directory_components = (
            split_root_components_from_filesystem_anchor(raw_root_path)
        )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current_directory_descriptor = os.open(filesystem_anchor, directory_flags)
        traversed_components: list[str] = []
        try:
            for directory_component in directory_components:
                traversed_components.append(directory_component)
                try:
                    next_directory_descriptor = os.open(
                        directory_component,
                        directory_flags,
                        dir_fd=current_directory_descriptor,
                    )
                except FileNotFoundError:
                    # note (machengqian): non-existent trailing target directories are valid before creation unless dangling symlinks.
                    try:
                        dangling_stat = os.lstat(
                            directory_component,
                            dir_fd=current_directory_descriptor,
                        )
                        if stat.S_ISLNK(dangling_stat.st_mode):
                            raise ValueError(
                                f"Symlink component {Path(filesystem_anchor, *traversed_components)} is not permitted in root path: {raw_root_path}"
                            ) from None
                        else:
                            pass
                    except FileNotFoundError:
                        pass
                    break
                else:
                    os.close(current_directory_descriptor)
                    current_directory_descriptor = next_directory_descriptor
        except OSError as os_error:
            if os_error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ValueError(
                    f"Symlink or non-directory component {Path(filesystem_anchor, *traversed_components)} is not permitted in root path: {raw_root_path}"
                ) from None
            else:
                raise
        finally:
            os.close(current_directory_descriptor)
        return Path(filesystem_anchor, *directory_components)


def parse_safe_relative_posix_parts(relative_posix_path: str) -> tuple[str, ...]:
    if "\\" in relative_posix_path or "\x00" in relative_posix_path:
        raise ValueError(f"Unsafe relative path: {relative_posix_path!r}")
    else:
        parsed_posix_path = PurePosixPath(relative_posix_path)
        if (
            parsed_posix_path.is_absolute()
            or not parsed_posix_path.parts
            or any(
                path_segment in ("", ".", "..")
                for path_segment in parsed_posix_path.parts
            )
        ):
            raise ValueError(f"Unsafe relative path: {relative_posix_path!r}")
        else:
            return tuple(parsed_posix_path.parts)


def open_directory_nofollow_chain(
    root_directory: Path, directory_parts: tuple[str, ...], *, create_parents: bool
) -> tuple[int, Path]:
    """Acquire root_directory from '/' via O_NOFOLLOW and traverse directory_parts descriptor-relatively."""
    current_directory_descriptor, verified_root = open_root_directory_nofollow(
        root_directory, create_if_missing=create_parents
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        for directory_segment in directory_parts:
            if create_parents:
                try:
                    os.mkdir(
                        directory_segment,
                        0o755,
                        dir_fd=current_directory_descriptor,
                    )
                except FileExistsError:
                    pass
            else:
                pass
            next_directory_descriptor = os.open(
                directory_segment,
                directory_flags,
                dir_fd=current_directory_descriptor,
            )
            os.close(current_directory_descriptor)
            current_directory_descriptor = next_directory_descriptor
    except OSError as os_error:
        os.close(current_directory_descriptor)
        if os_error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(
                f"Symlink or non-directory component encountered under {verified_root}"
            ) from None
        else:
            raise
    except BaseException:
        os.close(current_directory_descriptor)
        raise
    return current_directory_descriptor, verified_root


def ensure_safe_contained_path(
    root_directory: Path,
    relative_posix_path: str,
    *,
    create_parents: bool = False,
) -> Path:
    """Resolve a POSIX relative path under root_directory using descriptor-relative O_NOFOLLOW checks."""
    relative_parts = parse_safe_relative_posix_parts(relative_posix_path)
    parent_directory_descriptor, verified_root = open_directory_nofollow_chain(
        root_directory, relative_parts[:-1], create_parents=create_parents
    )
    try:
        try:
            leaf_stat = os.lstat(relative_parts[-1], dir_fd=parent_directory_descriptor)
            if stat.S_ISLNK(leaf_stat.st_mode):
                raise ValueError(
                    f"Symlink path components are not permitted: {relative_posix_path!r}"
                )
            else:
                pass
        except FileNotFoundError:
            pass
    finally:
        os.close(parent_directory_descriptor)
    return Path(verified_root, *relative_parts)


def read_bytes_nofollow(root_directory: Path, relative_posix_path: str) -> bytes:
    """Read a regular file under root_directory non-blockingly via dir_fd and O_NOFOLLOW."""
    relative_parts = parse_safe_relative_posix_parts(relative_posix_path)
    parent_directory_descriptor, _ = open_directory_nofollow_chain(
        root_directory, relative_parts[:-1], create_parents=False
    )
    try:
        try:
            # note (machengqian): O_NONBLOCK prevents writerless FIFOs or special files from hanging before fstat checks S_ISREG.
            file_descriptor = os.open(
                relative_parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_directory_descriptor,
            )
        except OSError as os_error:
            if os_error.errno == errno.ELOOP:
                raise ValueError(
                    f"Symlink path components are not permitted: {relative_posix_path!r}"
                ) from None
            elif os_error.errno == errno.ENXIO:
                raise ValueError(
                    f"Expected regular file at {relative_posix_path!r}"
                ) from None
            else:
                raise
        with os.fdopen(file_descriptor, "rb") as input_file:
            file_stat = os.fstat(input_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"Expected regular file at {relative_posix_path!r}")
            else:
                current_status_flags = fcntl.fcntl(input_file.fileno(), fcntl.F_GETFL)
                fcntl.fcntl(
                    input_file.fileno(),
                    fcntl.F_SETFL,
                    current_status_flags & ~os.O_NONBLOCK,
                )
                return input_file.read()
    finally:
        os.close(parent_directory_descriptor)


def unlink_contained_file_nofollow(
    root_directory: Path, relative_posix_path: str
) -> bool:
    """Unlink a regular file under root_directory using dir_fd and O_NOFOLLOW."""
    relative_parts = parse_safe_relative_posix_parts(relative_posix_path)
    try:
        parent_directory_descriptor, _ = open_directory_nofollow_chain(
            root_directory, relative_parts[:-1], create_parents=False
        )
    except FileNotFoundError:
        return False

    leaf_filename = relative_parts[-1]
    try:
        try:
            existing_stat = os.lstat(leaf_filename, dir_fd=parent_directory_descriptor)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(existing_stat.st_mode):
            raise ValueError(f"Refusing to unlink symlink at {relative_posix_path!r}")
        elif stat.S_ISDIR(existing_stat.st_mode):
            raise ValueError(f"Refusing to unlink directory at {relative_posix_path!r}")
        else:
            os.unlink(leaf_filename, dir_fd=parent_directory_descriptor)
            return True
    finally:
        os.close(parent_directory_descriptor)


def write_bytes_atomic(
    root_directory: Path, relative_posix_path: str, payload_bytes: bytes
) -> Path:
    """Atomically write bytes under root_directory using dir_fd and O_NOFOLLOW without following leaf symlinks."""
    relative_parts = parse_safe_relative_posix_parts(relative_posix_path)
    leaf_filename = relative_parts[-1]
    parent_directory_descriptor, verified_root = open_directory_nofollow_chain(
        root_directory, relative_parts[:-1], create_parents=True
    )
    try:
        try:
            existing_stat = os.lstat(leaf_filename, dir_fd=parent_directory_descriptor)
            if stat.S_ISLNK(existing_stat.st_mode):
                raise ValueError(
                    f"Refusing to overwrite symlink at {relative_posix_path!r}"
                )
            elif stat.S_ISDIR(existing_stat.st_mode):
                raise ValueError(
                    f"Refusing to overwrite directory at {relative_posix_path!r}"
                )
            else:
                pass
        except FileNotFoundError:
            pass

        nonce_hex = os.urandom(8).hex()
        temporary_leaf_filename = f".{leaf_filename}.{os.getpid()}.{nonce_hex}.tmp"
        temporary_file_descriptor = os.open(
            temporary_leaf_filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent_directory_descriptor,
        )
        try:
            with os.fdopen(temporary_file_descriptor, "wb") as output_file:
                output_file.write(payload_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())
            try:
                recheck_stat = os.lstat(
                    leaf_filename, dir_fd=parent_directory_descriptor
                )
                if stat.S_ISLNK(recheck_stat.st_mode):
                    raise ValueError(
                        f"Refusing to overwrite symlink at {relative_posix_path!r}"
                    )
                else:
                    pass
            except FileNotFoundError:
                pass
            os.replace(
                temporary_leaf_filename,
                leaf_filename,
                src_dir_fd=parent_directory_descriptor,
                dst_dir_fd=parent_directory_descriptor,
            )
        except BaseException:
            try:
                os.unlink(temporary_leaf_filename, dir_fd=parent_directory_descriptor)
            except FileNotFoundError:
                pass
            raise
    finally:
        os.close(parent_directory_descriptor)

    return Path(verified_root, *relative_parts)


def write_text_atomic(
    root_directory: Path, relative_posix_path: str, text_content: str
) -> Path:
    return write_bytes_atomic(
        root_directory, relative_posix_path, text_content.encode("utf-8")
    )


def write_json_atomic(destination_path: Path, json_payload: JsonValue) -> None:
    expanded_destination = destination_path.expanduser()
    serialized_json_bytes = (
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(
        expanded_destination.parent, expanded_destination.name, serialized_json_bytes
    )


def write_contained_json_atomic(
    root_directory: Path, relative_posix_path: str, json_payload: JsonValue
) -> Path:
    serialized_json_bytes = (
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return write_bytes_atomic(
        root_directory, relative_posix_path, serialized_json_bytes
    )
