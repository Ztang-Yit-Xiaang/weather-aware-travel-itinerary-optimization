# E3.UX1 Product Design Contract

**Status:** `verified`  
**Applies to:** product dashboard version `1.0.0`  
**Compatibility boundary:** read-only presentation of canonical run artifacts

## Information architecture

Customer mode answers, in order:

1. what happened and whether the repair is eligible;
2. the accepted itinerary and selected day;
3. the recommended result, changed/unchanged scope, permission impact, and
   tradeoffs;
4. parent/child metrics with direction and truth owner;
5. certificate and grounded explanation;
6. contextual map and its text alternative.

Research mode preserves the same selection and adds method identity, exact
refusal reasons, planner attempts, lineage, certificate details, `PlanDiff`,
route identities, canonical artifact paths, and full SHA-256 values.

## Layout contract

- At 1080px and above, use a three-region review layout: timeline, contextual
  map, and repair/evidence panel, followed by comparison and research evidence.
- From 720px to 1079px, use one reading column with two-column subgrids where
  content permits.
- Below 720px, follow issue → itinerary → repair → comparison → evidence → map.
  The 260px map is after the evidence and cannot consume the first viewport.
- Comparison tables may scroll inside their own wrapper; the document itself
  must never scroll horizontally.
- All permanent map labels are avoided. The layer control is compact, day
  selection is synchronized, and the map has a prose alternative.

## Design tokens

| Token role | Value | Use |
|---|---|---|
| Ink | `#17211b` | Primary text and primary action |
| Muted | `#59645c` | Secondary text |
| Paper | `#f4f1e8` | Page background |
| Surface | `#fffdf7` | Review panels |
| Teal | `#0b6f68` | Eligible/current state |
| Amber | `#854306` | Affected/incomplete warning state |
| Red | `#a63a32` | Failure/ineligible state |
| Blue | `#245f87` | Informational/read-only state |
| Focus | `#0b5fff` | Keyboard focus outline |
| Spacing | 4, 8, 12, 16, 24, 32px | Consistent rhythm |
| Radius | 6 and 12px | Controls and panels |

Status never relies on color alone: every chip has text, method cards have
status labels, and changed days have words as well as border treatment.

## Accessibility contract

- Native `header`, `main`, `section`, `footer`, button, table, `details`, and
  `summary` semantics.
- One `h1`, ordered section headings, explicit `aria-labelledby`, live status
  regions, pressed/current state attributes, and a skip link.
- Visible 3px focus treatment; 44×44px minimum interactive targets, including
  Leaflet zoom controls.
- Customer/research state and selected day remain stable across view changes.
- `prefers-reduced-motion` disables animation and smooth scrolling.
- Text alternative describes each itinerary day and marks affected days.
- The audited 390px snapshot has no contrast failures across 196 visible
  leaf-text nodes under WCAG AA size thresholds.
- Artifact text is inserted with `textContent`/`createElement`; the embedded
  data script escapes `<`, `>`, `&`, U+2028, and U+2029.

## Component boundaries

| Boundary | Responsibility |
|---|---|
| `product_dashboard_adapter.py` | Safe run-relative loading, content/lineage validation, finite JSON, source hashes, truth states |
| `product_dashboard_view_models.py` | Customer/research presentation data, metric ownership/direction, alternatives, map display geometry |
| `product_dashboard_assets.py` | Semantic HTML, tokenized CSS, read-only controller, map synchronization |
| `product_dashboard_renderer.py` | Non-overwritable export, product manifest, asset hashes, screenshot registration |
| Export/validation scripts | Create a derived snapshot and validate source/asset/security/compatibility contracts |

There is no frontend framework because the repository’s static artifact
architecture already supports the required accessibility, versioning, testing,
and rollback properties.

## Interaction boundary

The only primary action is **Review evidence**, which navigates within the
read-only artifact. No Accept, Keep, Ask permission, or Clarify control exists.
E3.UX5 remains `deferred`, disabled, E5-dependent, and outside this contract.

