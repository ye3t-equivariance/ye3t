Descriptor Enumeration And Label Printing
=========================================

The core enumerator lists canonical rank labels before any ACE chemistry or
radial basis is attached. It is useful for checking counts, printing compact
labels, and deciding which ranks should be exposed to a downstream descriptor
builder.

Exact enumeration works over orbit partitions of the repeated leaf labels. For
each rank ``nu``, ``enumerate_rank_labels`` returns records containing
``n_in``, ``l_in``, orbit partitions, and realized average angular momentum.

.. code-block:: python

   from ye3t import enumerate_rank_labels, format_ye3t_basis

   labels = enumerate_rank_labels(
       rank=4,
       target_l_avs=(0.0, 0.5, 1.0),
       strict_max_li=2,
       spec={"mode": "exhaustive"},
       max_labels=8,
   )

   for index, label in enumerate(labels):
       print(format_ye3t_basis(index, label))

Basis labels are schema-backed.  The public ``BasisLabel`` object can carry
fixed content, radial content, Young labels, angular paths, multiplicity
coordinates, conventions, and carrier type when those fields are known from the
compiled plan or exact basis metadata.  Use ``to_json`` for programmatic
serialization, ``to_latex`` for papers, ``to_human`` / ``format_ye3t_basis``
for reports, and ``to_filename_safe`` for artifacts.  Unknown fields remain
explicitly absent; formatters must not invent Young, LR, tableau, parity, or
convention data from a compact legacy tuple alone.

ACE descriptors add chemistry, radial channels, target angular momentum, and
optionally charge axes. Physical materialization and models live in the
separate ``ye3t-methods`` package; ``ye3t`` supplies their exact
representation-side labels and coupling coefficients.

Reusing enumeration and compiled maps
------------------------------------

Compact ``YE3TRepresentation`` inputs use ``eta_max_per_rank`` to bound the
number of distinct non-angular primitive identities in each product. Its
entries lie in ``1..N`` for factor rank ``N``. Species, the base radial
function and the source family identify a primitive; angular type and retained
role are separate inputs. A bound of one still permits every physical radial
function individually. Physical radial/angular dictionaries remain basis
settings. Existing ``eta_count_per_rank`` retains its historical capacity-hint
meaning and may be omitted when ``eta_max_per_rank`` is supplied.

The optional ``retained_max_per_rank`` has entries in ``0..N``. It bounds
singleton tags or occupied shell roles, excluding an explicit pooled
background. Total shell count remains a separate source setting. A zero bound
selects background-only shell occupations where that background exists;
shell-only sources then have an empty selected space. Complete shell-action
orbits and tableau/magnetic components remain present. Compiler catalogues
apply these bounds before their independent-copy inventory and coefficient
materialization. Explicit contents or occupations conflicting with a bound
raise an input error.

For abstract ``count_fixed_content``, ``content`` provides non-angular
primitive labels and ``input_Ls`` supplies their angular types. A restrictive
retained-role bound requires a source plan with explicit role identities;
ordinary ``ACE_density`` has zero retained roles. The existing canonical leaf
iterator and count accept ``eta_max`` for physical primitive dictionaries.

Use ``ye3t.couplings.count``, ``plan`` and ``compile`` for the selected carrier.

For general tagged-Cauchy scalar requests, the catalogue may contain
``chemical_patterns_by_rank``. Each rank maps to a list of neighbor chemical
multisets of that length, for example ``{4: [["H", "H", "O", "O"]]}``.
Omitted ranks retain all chemical choices within the other limits. The shared
compiler applies this selection before coefficient materialization. It does
not filter central species or change their independent coefficient blocks.
Raw count/plan reports are upper bounds until physical-image reduction has
resolved identities; compiled labels give the final independent coordinates.

Their existing mathematical caches retain valid labels, Young/angular blocks
and coupling paths. Tagged physical-coordinate maps and shell source maps also
use the shared store, so extending a catalogue can reuse smaller compiled
pieces. Applications do not need to enumerate their own labels or rerun the
compiler during energy and force evaluation.

The store defaults to ``~/.cache/ye3t`` (or the XDG cache directory), with
``YE3T_CACHE_DIR`` as an override. ``YE3T_CACHE_MODE`` selects ``auto``,
``read_only``, ``refresh`` or ``off``. Normal cache hits check stored integrity
and request bindings. Unchanged files have bounded process-local reuse, with
separate values returned to callers. General tagged-image independent validation
is retained across processes by content and validator revision.
Scalar parent maps shared by different catalogues reuse completed exact checks
within a process. Loading checks the current artifact contents before using
this bounded memo. Individual parent checks create no disk records; persistent
validation reuse operates on complete tagged images.

General tagged compilation also saves the complete independent physical image
under the full compiler plan. Repeating that plan reuses its assembled maps;
extending the catalogue can reuse the individual coordinate maps. Complete
images use losslessly compressed JSON in the same cache directory. Their
public labels, coefficient ordering and saved-model format remain unchanged.
In read-only mode, an absent complete image can still be assembled from
available coordinate maps without writing a new cache entry.

For an explicit audit, set ``YE3T_CACHE_VERIFY=full``. This replays the relevant
independent construction checks. A new compiler convention or validation rule
must update its cache schema or producer revision. Digests serve internal cache
and saved-model integrity; routine examples need only show the selected source,
feature counts and evaluated results.

The optional Triton runtime uses the same persistent cache root, under
``runtime/triton_cache``. Set ``YE3T_CACHE_DIR`` before importing the runtime to
choose that root; otherwise the XDG directory or ``~/.cache/ye3t`` is used.
Explicit ``TRITON_CACHE_DIR`` and existing runtime-directory overrides retain
precedence. Application runtimes use the same default, regardless of import
order. Triton retains its own compiler/device compatibility keys; this does
not bypass coefficient validation or change a model's basis. Kernel storage
is separate from ``YE3T_CACHE_MODE``, which controls coupling artifacts.
An upgrade from the historical temporary default may compile kernels once
in the new location. Subsequent processes can reuse them. Saved models do
not require distributing the machine-specific kernel cache.

Coupling Namespace
------------------

Valid descriptor labels, multiplicity counts, backend plans, and coupling
coefficients should be requested through ``ye3t.couplings``:

.. code-block:: python

   from ye3t.couplings import count, plan, compile_ace_factorized_schedules_by_L

   report = count(content=(1, 1, 2), input_Ls=(1, 1, 0), target_L=0)
   labels = report.labels_for_target(0)
   report.require_label(labels[0], target_L=0)

   coupler_plan = plan(report)
   compiled = compile_ace_factorized_schedules_by_L(
       content=report.content, input_Ls=(1, 1, 0),
   )
   assert compiled.schedules_by_L[0].basis_count == len(labels)

``count`` is the public fixed-content multiplicity report.  ``plan`` records
the backend choice and count provenance before coefficient construction.
The factorized ACE compiler materializes practical coupling schedules and
attaches a validation report. Downstream packages such as
``ye3t-methods`` may materialize descriptors from these reports and plans, but
they should not decide locally which symmetry labels are valid.

For nonzero angular factors in ordinary density,
``compile(..., allow_dense_reference=True)`` can check the full
``(content, l)`` block decomposition and every angular path. It projects the
globally trivial Young output onto commutative density and verifies that the
compact ACE labels span the same physical space with one common change of
basis across all magnetic components. The compact coefficients remain the
evaluation and serialization convention. This joint reference compilation
has a bounded dense-table size cap and is meant for tests and comparisons;
``count`` and ``plan`` do not materialize the table.

The public cached numerical Specht materializer uses the same canonical
binary split maps and actual intermediate subgroup labels as the exact
constructor. Its versioned coefficient-array cache preserves the selected
copy convention. Historical nullspace cache files remain separate; they
retain their original coordinates. Ordinary construction records trusted
constructor metadata, while explicit diagnostics perform residual checks.

For an ordinary commutative ACE-density label, ``compile_ace_coordinate``
materializes all ``2L+1`` complex-magnetic components. It composes the
compiler's symmetric-power block coordinates with its angular schedule,
then collects equivalent density monomials. The returned validation record checks
fixed-content membership, magnetic sign reversal, and the SO(3) raising
generator. The algebraic map is exact for compiler-issued labels with maximal
fixed-content blocks through rank eight; stored binary64 coefficients use
tolerance pruning and pass numerical validation. ``L=0`` delegates to the
existing scalar compiler so its coefficient convention stays compatible.

.. code-block:: python

   from ye3t.couplings import count, compile_ace_coordinate

   report = count(content=(1, 1, 1), input_Ls=(1, 1, 1),
                  target_L=1, target_permutation="trivial",
                  carrier="ACE_density")
   coordinate = compile_ace_coordinate(report.labels_for_target(1)[0])
   table = coordinate["coefficient_table"]
   print(table.M_R_values.tolist())  # [-1, 0, 1]
