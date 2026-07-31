# IDS uEye runtime (for the uEye XS camera type)

The four Debian packages the `UEyeXS` driver needs at runtime:

| package | why |
|---|---|
| `ueye-api` | `libueye_api.so` — what `pyueye` wraps |
| `ueye-common` | shared files the api package depends on |
| `ueye-driver-usb` | the `ueyeusbdrc` daemon that claims USB uEye cameras and uploads their firmware |
| `ueye-tools-cli` | command-line utilities (diagnostics) |

Installed **automatically and idempotently by the upgrade** (see the
`camera/setup.sh` sub-setup on the upgrade repo's `vision_pro` branch):
skipped when this exact version is already installed, re-run when these
files change. Harmless on units with no uEye plugged in — the daemon
idles and D405 operation is unaffected.

Source: "IDS Software Suite 4.96.1 for ARMv8 64-bit (hf) — Debian
package" from ids-imaging.com (login-gated download; that's why the debs
are vendored here instead of fetched at upgrade time). The full suite
also ships GigE drivers, manuals, demos and Qt tools — not needed, not
vendored. NOTE: IDS software is proprietary — keep this repo private.
