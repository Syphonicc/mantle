# Contributing to mantle

Thanks for your interest! mantle is an early-stage research mesher, so the most valuable contributions right now are:

1. **Test geometries.** Public STL files that break current meshers (or mantle). Open a *Geometry / benchmark request* issue.
2. **Bug reports** with the summary mantle prints and the failing `checkMesh` lines.
3. **Code**: see the open issues and the roadmap in the README.

## Building and testing

```bash
cd rcore
cargo build --release
cargo test --release      # must pass, and the build must have 0 warnings
```

The Rust core has **no external dependencies** (std only); please keep it that way unless discussed in an issue first.

## Conventions

- Keep changes focused; one topic per pull request.
- New Python script versions get a new filename (`*_v2.py`), so earlier results stay reproducible.
- Every claim about mesh quality should come with numbers: `checkMesh` verdict, non-positive cell count, minimum scaled Jacobian, and, where relevant, a solver run (OpenFOAM or SU2).
- Generated meshes and solver output stay out of git (see `.gitignore`).

## License

By contributing you agree that your contributions are licensed under the [Apache-2.0](LICENSE) license.
