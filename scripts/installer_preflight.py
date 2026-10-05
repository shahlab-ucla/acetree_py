"""Check interpreter compatibility before either native installer invokes pip."""

import platform
import sys


def compatibility_error(variant, version_info, system, machine):
    """Return actionable guidance for an unsupported installation target."""
    if tuple(version_info[:2]) < (3, 10):
        return "AceTree-Py requires Python 3.10 or newer."
    if (
        variant != "core"
        and tuple(version_info[:2]) >= (3, 14)
        and system == "Darwin"
        and machine.lower() in {"x86_64", "amd64", "i386", "i686"}
    ):
        return (
            "AceTree-Py GUI installation on Intel macOS with Python 3.14 or newer "
            "is unsupported. Use Python 3.10-3.13 for the GUI, or select the core "
            "variant for a core-only installation."
        )
    return None


def main():
    error = compatibility_error(
        sys.argv[1], sys.version_info, platform.system(), platform.machine()
    )
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
