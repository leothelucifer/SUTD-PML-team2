# Final presentation and design

| File | What it is |
|---|---|
| `sepsis-early-warning.pptx` | Final presentation, 22 slides, 16:9, with speaker notes. Each slide is a full-slide image rendered from the source below. |
| `source/deck.json`, `source/slides/*.html` | Editable slide source (one HTML file per slide, in order given by `deck.json`). |
| `assets/` | Images used by the slides: AWS Architecture Icons and three aggregate figures from the hospital comparison (PR #48). `blob-map.json` maps the asset ids in the slide HTML to these files. |
| `tools/render_slides.js`, `tools/build_pptx.py` | Rebuild the PowerPoint from the source. |
| `../design/sepsis-warning-pipeline.html` | Design page: data split, training, model selection, alert policy, ICU call contract, AWS deployment and MLOps loop. Open it in a browser. |

## Rebuild the PowerPoint after editing a slide

```bash
node docs/presentation/tools/render_slides.js /tmp/deck-render
python docs/presentation/tools/build_pptx.py /tmp/deck-render docs/presentation/sepsis-early-warning.pptx
```

`render_slides.js` needs Playwright with Chromium and network access to Google Fonts (Inter, Space Grotesk). `build_pptx.py` needs `python-pptx`, which is not a project dependency.

The slide text in the `.pptx` is part of the image, so text edits belong in `source/slides/*.html`, followed by a rebuild. The deck was authored in a Claude Slides artifact; an editable `.pptx` can also be exported from there (Share → Export) and committed in place of this file.

## Still to fill in

Bracketed placeholders (`[__]`) mark results that need real runs: model comparison, hospital A vs B, HMM ablation, SHAP drivers, drift report, fairness gaps, latency and the presentation date.

## Slide owners

| Owner | Slides |
|---|---|
| Edwin | 4 Data and split · 5 Bedside to model · 9 EDA · 10 SIRS baseline |
| Jawad | 6 Pipeline · 7 Causal features |
| Kaiyuan | 3 Objective · 8 Modelling ladder · 11 Score to alert · 12 Model comparison |
| Josh | 14 HMM states and patient groups |
| Leo | 16 Building blocks · 17 Commit to production · 18 AWS deployment |
| Mashie | 13 Hospital B test · 15 Explanations · 19 Monitoring · 20 Governance |

## Attribution

AWS service icons are from the AWS Architecture Icons set, used to depict AWS services in architecture diagrams. Data: PhysioNet/CinC Challenge 2019 (Reyna et al., Crit Care Med 2020), CC BY 4.0. All figures are aggregates; no patient-level data is included.
