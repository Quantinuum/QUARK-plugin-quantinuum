# Development

## Publishing to PyPI

Publishing is handled by the GitHub Actions workflow in `.github/workflows/release.yml`.

Pull requests and pushes to `main` build the sdist and wheel with `uv build` to validate packaging.

To publish:

1. Configure this GitHub repository as a trusted publisher for the `QUARK-plugin-quantinuum` project on PyPI.
2. Create a GitHub Release for the version you want to publish.
3. When the release is published, the workflow will build the sdist and wheel from the release tag and upload them to PyPI.

## Versioning

Package versions are derived from git tags via `setuptools-scm`.

- A release created from tag `v0.1.0` will publish package version `0.1.0`.
- Development builds created from commits after a tag will get an auto-generated development version.

Each GitHub Release should therefore point at a clean version tag such as `v0.1.0`.
