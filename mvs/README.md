# Hikrobot MVS (for the HikRobot camera type)

The `HikRobot` driver talks to GigE (PoE) cameras through Hikrobot's
MVS SDK. Everything the fleet needs is vendored here (same pattern as
the IDS debs in `ids/`):

| file | what |
|---|---|
| `MvImport/` | the MVS **Python bindings** (ctypes wrappers), extracted from the deb below. Pure Python, cross-platform: on Windows they load `MvCameraControl.dll` from PATH, on Linux `libMvCameraControl.so` via the `MVCAM_COMMON_RUNENV` env var the installer sets. The driver searches this folder FIRST — no path guessing on any machine. |
| `MVS-5.0.2_aarch64_20260728.deb.part-aa/-ab` | the MVS **Linux runtime** for the Pis (aarch64), split in two because GitHub rejects files over 100 MB (the deb is 109.5 MB) |

## Install on a Pi

```bash
cat mvs/MVS-5.0.2_aarch64_20260728.deb.part-* > /tmp/MVS.deb
sudo dpkg -i /tmp/MVS.deb
rm /tmp/MVS.deb
```

Reassembled-deb integrity check (must print this exact hash):

```
$ cat mvs/MVS-5.0.2_aarch64_20260728.deb.part-* | sha256sum
21d9f536b034ef01af8fed24370597d9f5aca16c1e0235e36614e30d7dafb301
```

## Install on a Windows PC

The vendored bindings work, but the runtime DLL must come from the MVS
Windows installer (hikrobotics.com → Download → "Machine Vision
Software MVS (Windows)", same 5.0.2 version) — it is ~400 MB and a
one-time dev-PC install, so it is deliberately NOT vendored. Restart
any Python kernel after installing so the new PATH is seen.

## Notes

- Source: "Machine Vision Software MVS V5.0.2 (Linux)" from
  hikrobotics.com (verification-gated download — that's why the files
  are vendored instead of fetched at setup time).
- Keep the bindings and installed runtimes on the SAME MVS version
  (5.0.2 everywhere) — mixed versions are an unsupported combination.
- MVS is proprietary — keep this repo private (same rule as `ids/`).
