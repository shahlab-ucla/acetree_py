#!/usr/bin/env sh
set -eu

EXPECTED_BRANCH="subcellular-measurements"
VARIANT="gui"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--variant core|gui|all] [--python PATH] [--dry-run]" \
        "" \
        "Installs the subcellular-measurements AceTree-Py checkout in editable mode."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)
            [ "$#" -ge 2 ] || { printf '%s\n' "--variant requires a value" >&2; exit 2; }
            VARIANT="$2"
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || { printf '%s\n' "--python requires a value" >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '%s\n' "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$VARIANT" in
    core|gui|all) ;;
    *)
        printf '%s\n' "Invalid variant '$VARIANT'; choose core, gui, or all." >&2
        exit 2
        ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if [ -e "$REPO_ROOT/.git" ]; then
    CURRENT_BRANCH=$(git -C "$REPO_ROOT" branch --show-current)
    if [ -z "$CURRENT_BRANCH" ]; then
        printf '%s\n' "Detached checkout: switch to branch '$EXPECTED_BRANCH' before installing." >&2
        exit 1
    elif [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
        printf '%s\n' \
            "This installer requires branch '$EXPECTED_BRANCH'; current branch is '$CURRENT_BRANCH'." \
            "Run: git switch $EXPECTED_BRANCH" >&2
        exit 1
    fi
fi

if [ "$VARIANT" = "core" ]; then
    INSTALL_TARGET="$REPO_ROOT"
else
    INSTALL_TARGET="${REPO_ROOT}[$VARIANT]"
fi

printf '%s\n' \
    "Installing AceTree-Py '$VARIANT' from branch '$EXPECTED_BRANCH'." \
    "$PYTHON_BIN -m pip install --editable \"$INSTALL_TARGET\""

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

"$PYTHON_BIN" -m pip install --editable "$INSTALL_TARGET"
VERSION_OUTPUT=$("$PYTHON_BIN" -m acetree_py --version)
case "$VERSION_OUTPUT" in
    *"(tracking integration)"*) ;;
    *)
        printf '%s\n' "The installed build did not identify itself as the tracking integration." >&2
        exit 1
        ;;
esac
printf '%s\n' "$VERSION_OUTPUT"
