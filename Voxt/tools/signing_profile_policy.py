"""Validate explicit profile-backed Data Protection identity for local packaging."""

from __future__ import annotations

import argparse
import datetime
import plistlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

APPLICATION_IDENTIFIER = "com.apple.application-identifier"
TEAM_IDENTIFIER = "com.apple.developer.team-identifier"


def plist_dictionary(payload: bytes, label: str) -> dict[str, object]:
    decoded: object = plistlib.loads(payload)
    if not isinstance(decoded, dict) or not all(
        isinstance(name, str) for name in decoded
    ):
        raise ValueError(f"{label} must be a plist dictionary")
    else:
        return decoded


def string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(text, str) and text for text in value
    ):
        raise ValueError(f"{label} must be a nonempty string array")
    else:
        return value


def certificate_list(value: object) -> list[bytes]:
    if not isinstance(value, list) or not value or not all(
        isinstance(certificate, bytes) and certificate for certificate in value
    ):
        raise ValueError("profile requires DeveloperCertificates DER data")
    else:
        return value


def prepare_entitlements(
    profile: dict[str, object], source: dict[str, object], bundle_identifier: str
) -> dict[str, str | bool | list[str]]:
    """Resolve only profile-authorized private app identity, never a new group."""
    now = datetime.datetime.now(datetime.timezone.utc)
    expiration = profile.get("ExpirationDate")
    creation = profile.get("CreationDate")
    if not isinstance(expiration, datetime.datetime) or not isinstance(
        creation, datetime.datetime
    ):
        raise ValueError("profile requires CreationDate and ExpirationDate")
    elif not (
        creation.replace(tzinfo=datetime.timezone.utc)
        <= now < expiration.replace(tzinfo=datetime.timezone.utc)
    ):
        raise ValueError("profile is expired or not yet valid")
    else:
        pass
    if string_list(profile.get("Platform"), "profile Platform") != ["OSX"]:
        raise ValueError("profile must target macOS (OSX)")
    elif profile.get("ProvisionsAllDevices") is not True or "ProvisionedDevices" in profile:
        raise ValueError("profile must be a Developer ID distribution profile, not development or ad hoc")
    else:
        pass
    teams = string_list(profile.get("TeamIdentifier"), "profile TeamIdentifier")
    prefixes = string_list(
        profile.get("ApplicationIdentifierPrefix"), "profile ApplicationIdentifierPrefix"
    )
    if len(teams) != 1 or len(prefixes) != 1 or not all(
        re.fullmatch(r"[A-Z0-9]{10}", identity) for identity in [*teams, *prefixes]
    ):
        raise ValueError("profile requires one explicit team and application prefix")
    else:
        pass
    certificate_list(profile.get("DeveloperCertificates"))
    authorized = profile.get("Entitlements")
    if not isinstance(authorized, dict):
        raise ValueError("profile requires an Entitlements dictionary")
    elif authorized.get("get-task-allow", False) is not False:
        raise ValueError("development entitlements are not permitted")
    else:
        pass
    app_identity = f"{prefixes[0]}.{bundle_identifier}"
    if authorized.get(APPLICATION_IDENTIFIER) != app_identity or "*" in app_identity or "$(" in app_identity:
        raise ValueError("profile application identifier must match the explicit app bundle identifier; wildcard profiles are not permitted")
    elif authorized.get(TEAM_IDENTIFIER) != teams[0]:
        raise ValueError("profile entitlement team does not match TeamIdentifier")
    else:
        pass
    authorized_groups = string_list(
        authorized.get("keychain-access-groups"), "profile keychain-access-groups"
    )
    # note (guanwei): A profile prefix allowlist authorizes the explicit private group, not a wildcard signature.
    if app_identity not in authorized_groups and f"{prefixes[0]}.*" not in authorized_groups:
        raise ValueError("profile does not authorize the app's private keychain group")
    else:
        pass
    replacements = {
        "$(AppIdentifierPrefix)": f"{prefixes[0]}.",
        "$(DEVELOPMENT_TEAM)": teams[0],
        "$(PRODUCT_BUNDLE_IDENTIFIER)": bundle_identifier,
    }
    expanded: dict[str, str | bool | list[str]] = {}
    for name, value in source.items():
        if isinstance(value, bool):
            expanded[name] = value
        else:
            texts = (
                [value] if isinstance(value, str) else string_list(value, f"source {name}")
            )
            resolved: list[str] = []
            for text in texts:
                for token, replacement in replacements.items():
                    text = text.replace(token, replacement)
                if "$(" in text:
                    raise ValueError(f"unresolved signing variable in {name}")
                else:
                    resolved.append(text)
            expanded[name] = resolved[0] if isinstance(value, str) else resolved
    if expanded.get(APPLICATION_IDENTIFIER) != app_identity:
        raise ValueError("source application identifier differs from the authorized app identity")
    elif expanded.get(TEAM_IDENTIFIER) != teams[0]:
        raise ValueError("source team differs from the authorized team")
    elif expanded.get("keychain-access-groups") != [app_identity]:
        raise ValueError("source default keychain group must equal the authorized app identity")
    else:
        return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-entitlements", type=Path, required=True)
    parser.add_argument("--info-plist", type=Path, required=True)
    parser.add_argument("--output-entitlements", type=Path)
    parser.add_argument("--signed-app", type=Path)
    arguments = parser.parse_args()
    if (arguments.output_entitlements is None) == (arguments.signed_app is None):
        parser.error("supply exactly one of --output-entitlements or --signed-app")
    else:
        pass
    decoded_profile = subprocess.run(
        ["security", "cms", "-D", "-i", str(arguments.profile)],
        check=True, capture_output=True,
    )
    profile = plist_dictionary(decoded_profile.stdout, "profile")
    source = plist_dictionary(
        arguments.source_entitlements.read_bytes(), "source entitlements"
    )
    metadata = plist_dictionary(arguments.info_plist.read_bytes(), "app Info.plist")
    bundle_identifier = metadata.get("CFBundleIdentifier")
    if not isinstance(bundle_identifier, str) or not re.fullmatch(
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", bundle_identifier
    ):
        raise ValueError("app requires an explicit CFBundleIdentifier")
    else:
        pass
    expanded = prepare_entitlements(profile, source, bundle_identifier)
    if arguments.signed_app is None:
        arguments.output_entitlements.write_bytes(plistlib.dumps(expanded))
    else:
        signature = subprocess.run(
            ["codesign", "--display", "--verbose=4", str(arguments.signed_app)],
            check=True, capture_output=True, text=True,
        )
        signature_lines = signature.stderr.splitlines()
        if f"TeamIdentifier={expanded[TEAM_IDENTIFIER]}" not in signature_lines or not any(
            line.startswith("Authority=Developer ID Application:") for line in signature_lines
        ):
            raise ValueError("signature must use Developer ID Application and the profile's team")
        else:
            pass
        signed_entitlements = subprocess.run(
            ["codesign", "--display", "--entitlements", ":-", str(arguments.signed_app)],
            check=True, capture_output=True,
        )
        if plist_dictionary(signed_entitlements.stdout, "signed entitlements") != expanded:
            raise ValueError("signed entitlements differ from the authorized source entitlements")
        else:
            pass
        with tempfile.TemporaryDirectory(prefix="voxt-signing-certificate-") as directory:
            certificate_prefix = Path(directory) / "certificate"
            subprocess.run(
                ["codesign", "--display", f"--extract-certificates={certificate_prefix}",
                 str(arguments.signed_app)],
                check=True, capture_output=True,
            )
            if Path(f"{certificate_prefix}0").read_bytes() not in certificate_list(
                profile.get("DeveloperCertificates")
            ):
                raise ValueError("signing certificate is not authorized by the supplied profile")
            else:
                pass


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(
            f"error: signing profile policy: {error.cmd[0]} failed with status {error.returncode}",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode) from error
    except (ValueError, OSError, plistlib.InvalidFileException) as error:
        raise SystemExit(f"error: signing profile policy: {error}") from error
    else:
        pass
else:
    pass
