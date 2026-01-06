This directory contains git patch files downloaded from GitHub and, in particular, from Pull Requests to the
`synthcity` repository: https://github.com/vanderschaarlab/synthcity/tree/main

### What are these patch files?
First of all, a git patch file is a text file that represents changes between two sets of files or commits.
Read more here: https://graphite.com/guides/git-apply-patch. The patch files in this directory correspond to
the following Pull Requests, both of which are raised by community members using the `synthcity` package:
- https://github.com/vanderschaarlab/synthcity/pull/351; and
- https://github.com/vanderschaarlab/synthcity/pull/353.

These Pull Requests contribute important source code. In particular:
- [Synthcity#351](https://github.com/vanderschaarlab/synthcity/pull/351) fixes a bug in the PrivBayes generator. 
This is important for the soundness of our findings, since we are using this generator in our investigation; and
- [Synthcity#353](https://github.com/vanderschaarlab/synthcity/pull/353) enables the usage of `torch` version 2.3+
and Python 3.9+. This resolves compatibility issues with the rest of the generators used in our investigation,
e.g., GReaT and REaLTabFormer.

### Why do we need these patch files?

We need these patch files, because, as of today, Jan 06 2026, these Pull Requests **are not yet accepted nor merged**
into the main `synthcity` package. In other words, as of now, when installing `pip install synthcity`, these fixes are
not included. For this reason, we have opted for the most reproducible mechanism to install the (base) `syncthcity`
package _plus_ the fixes. This mechanism is implemented in `install-revamped-synthcity.sh`, making it reproducible by
anyone who can run `git` and `uv` (or `pip`) commands now or any time in the future.

### What would be other alternatives to achieve the same result?
1. **Wait untill the Pull Requests are merged**. Once these Pull Requests are merged into the main branch of 
`synthcity`, the fixes will be available in the next release after the merge date. We discarded this alternative
because there was no guarantee that the merging will happen soon or even happen. The Pull Requests have been open for 
3 weeks and 2 months respectively (as of Jan 06, 2026) without any comment or action from the `synthcity` authors.
2. **Use `git merge` operations instead of patch files**. We could have avoided using patch files by fetching the 
current main branch and the two separate branches (i.e., the two Pull Requests) and merging them using `git`.
In this case, behind the scenes, `git` would use the same patch files, abstracting them from the end user. We 
discarded this alternative because we could not guarantee reproducibility in case the Pull Requests were merged and
the separate branches were deleted by the contributors. In this case, the process would fail during the step of
fetching the separate branches to merge into the main one.

### Reproducibility Guarantee

All in all, as long as these patch files live inside this repo, we can guarantee reproducibility of our `syncthcity`
installation. This is crucial, since a large amount of the generators we use for our investigation are actually
sourced from this package.
