"""
insert_figures_to_docx.py
Insert the 6 hover evaluation figures into RL_Results_Mentor_Review_v6.docx
to produce RL_Results_Mentor_Review_v7.docx.

Prerequisites:
  - Run generate_hover_figures.py on NAS first.
  - Figures must be present at: FIGURES_DIR (see path constants below)

Run locally on Steam Deck (built-in Python 3, no extra packages):
    python3 insert_figures_to_docx.py
"""

import os
import struct
import zipfile
import shutil
import re

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.expanduser(
    "~/ThesisStorage/results/pyflyt/hover/figures"
)
INPUT_DOCX  = os.path.join(WORKSPACE, "RL_Results_Mentor_Review_v6.docx")
OUTPUT_DOCX = os.path.join(WORKSPACE, "RL_Results_Mentor_Review_v7.docx")

# ── Figure catalogue  (filename, caption) ──────────────────────────────────────
FIGURES = [
    ("fig1_far_comparison.png",
     "Figure 1. False Alarm Rate by detection channel in the none (baseline) "
     "condition across 10 seeds (600 k-step reset-free deployment). "
     "D_k achieves 0 % FAR; CUSUM 80 %; OCSVM 100 %.  "
     "Target threshold: FAR ≤ 5 %."),

    ("fig2_dk_trajectories.png",
     "Figure 2. D_k (JSD) trajectory per seed across the three drift conditions. "
     "Thin lines = individual seeds (blue = Q1, orange = Q2). "
     "Dashed horizontal line: \\u03c4_warn = 0.008 (tau_warn_floor). "
     "Red dotted vertical line (abrupt panel): T_drift = 315 k."),

    ("fig3_detection_matrix.png",
     "Figure 3. Detection outcome matrix for the D_k channel. "
     "Green = timely detection (\\u0394 = lead time in thousands of steps); "
     "dark-orange = pre-drift failure (policy collapsed before drift injection); "
     "amber = missed detection."),

    ("fig4_lead_times.png",
     "Figure 4. Detection lead time (\\u0394 = T_hard \\u2013 T_warn) for detected "
     "seeds only. Horizontal bar = mean. Gradual drift provides substantially "
     "greater warning than abrupt drift (mean 63.4 k vs 11.6 k steps)."),

    ("fig5_weibull_survival.png",
     "Figure 5. Weibull survival curves S(t) = exp(\\u2212(t/\\u03bb)\\u1d4f). "
     "Left: abrupt drift (Q1 k=31.2, \\u03bb=324.8 k; Q2 k=110.2, \\u03bb=327.0 k). "
     "Right: gradual drift (Q1 k=2.93, \\u03bb=125.1 k; Q2 k=8.83, \\u03bb=174.5 k). "
     "Rug marks = empirical T_hard; dotted line = S(t)=0.10 (RUL_90)."),

    ("fig6_alarm_timelines.png",
     "Figure 6. Per-seed alarm timeline for the abrupt drift condition. "
     "Orange triangle = T_ocsvm; purple square = T_cusum; "
     "green circle = T_warn (D_k); black cross = T_hard; "
     "red dashed line = T_drift (315 k). OCSVM fires pre-drift in all 10 seeds; "
     "CUSUM fires pre-drift in 3 seeds; D_k fires at or after T_drift."),
]


# ── PNG utilities ───────────────────────────────────────────────────────────────
def png_dimensions(path):
    """Return (width_px, height_px) from a PNG file header."""
    with open(path, "rb") as fh:
        sig = fh.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a PNG file.")
        fh.read(4)   # chunk length
        fh.read(4)   # "IHDR"
        w, h = struct.unpack(">II", fh.read(8))
    return w, h

DPI = 150
EMU_PER_INCH = 914400

def px_to_emu(px):
    return int(px * EMU_PER_INCH / DPI)

MAX_WIDTH_EMU  = int(5.5 * EMU_PER_INCH)   # 5.5 inches = content width with margins


# ── DOCX XML builders ───────────────────────────────────────────────────────────
NS_WP  = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A   = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS_R   = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

def drawing_xml(rel_id, cx_emu, cy_emu, desc, draw_id):
    """Build the <w:drawing> XML fragment for an inline image.
    All namespaces declared inline so the fragment is self-contained."""
    return (
        f'<w:drawing>'
        f'<wp:inline xmlns:wp="{NS_WP}" distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx_emu}" cy="{cy_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{draw_id}" name="Figure{draw_id}" descr="{desc}"/>'
        f'<wp:cNvGraphicFramePr>'
        f'<a:graphicFrameLocks xmlns:a="{NS_A}" noChangeAspect="1"/>'
        f'</wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{NS_A}">'
        f'<a:graphicData uri="{NS_PIC}">'
        f'<pic:pic xmlns:pic="{NS_PIC}">'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{draw_id}" name="Figure{draw_id}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill>'
        f'<a:blip xmlns:r="{NS_R}" r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx_emu}" cy="{cy_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData>'
        f'</a:graphic>'
        f'</wp:inline>'
        f'</w:drawing>'
    )

def image_paragraph_xml(rel_id, cx_emu, cy_emu, desc, draw_id):
    """Paragraph containing a centred image."""
    drw = drawing_xml(rel_id, cx_emu, cy_emu, desc, draw_id)
    return (
        f'<w:p>'
        f'<w:pPr><w:jc w:val="center"/>'
        f'<w:spacing w:before="240" w:after="60"/></w:pPr>'
        f'<w:r>{drw}</w:r>'
        f'</w:p>'
    )

def caption_paragraph_xml(caption_text):
    """Italic centred caption paragraph."""
    safe = (caption_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
    return (
        f'<w:p>'
        f'<w:pPr><w:jc w:val="center"/>'
        f'<w:spacing w:before="0" w:after="240"/></w:pPr>'
        f'<w:r><w:rPr><w:i/><w:sz w:val="18"/></w:rPr>'
        f'<w:t xml:space="preserve">{safe}</w:t>'
        f'</w:r>'
        f'</w:p>'
    )

def section_heading_xml(text):
    """Heading 2 paragraph for the figures section."""
    return (
        f'<w:p>'
        f'<w:pPr><w:pStyle w:val="Heading2"/>'
        f'<w:pageBreakBefore/>'
        f'<w:spacing w:before="240" w:after="120"/></w:pPr>'
        f'<w:r><w:t>{text}</w:t></w:r>'
        f'</w:p>'
    )

def page_break_xml():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


# ── Relationship helpers ─────────────────────────────────────────────────────────
REL_TYPE_IMAGE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
)

def find_max_rel_id(rels_xml):
    """Return the highest numeric rId found in the relationships XML."""
    ids = re.findall(r'Id="rId(\d+)"', rels_xml)
    return max((int(i) for i in ids), default=0)

def new_rel_entry(rel_id_str, media_path):
    return (
        f'<Relationship Id="{rel_id_str}" '
        f'Type="{REL_TYPE_IMAGE}" '
        f'Target="media/{media_path}"/>'
    )

def inject_rel(rels_xml, entry):
    """Insert a new <Relationship/> before </Relationships>."""
    return rels_xml.replace("</Relationships>", entry + "\n</Relationships>")

def inject_content_type(ct_xml, ext="png"):
    """Add PNG content-type default if not already present."""
    if f'Extension="{ext}"' in ct_xml:
        return ct_xml
    entry = f'<Default Extension="{ext}" ContentType="image/png"/>'
    return ct_xml.replace("</Types>", entry + "\n</Types>")


def main():
    # 1. Validate inputs
    missing = [f for f, _ in FIGURES
               if not os.path.isfile(os.path.join(FIGURES_DIR, f))]
    if missing:
        print("ERROR: The following figures have not been generated yet:")
        for m in missing:
            print(f"  {os.path.join(FIGURES_DIR, m)}")
        print("\nRun generate_hover_figures.py on the NAS first, then retry.")
        return

    print(f"Input:  {INPUT_DOCX}")
    print(f"Output: {OUTPUT_DOCX}")
    print(f"Figures: {FIGURES_DIR}\n")

    # 2. Copy input to output (we modify the copy)
    shutil.copy2(INPUT_DOCX, OUTPUT_DOCX)

    # 3. Open, modify, re-zip
    import io
    with zipfile.ZipFile(INPUT_DOCX, "r") as zin:
        names = zin.namelist()
        file_contents = {n: zin.read(n) for n in names}

    # ── Decode key XML files
    doc_xml  = file_contents["word/document.xml"].decode("utf-8")
    rels_xml = file_contents["word/_rels/document.xml.rels"].decode("utf-8")
    ct_xml   = file_contents["[Content_Types].xml"].decode("utf-8")

    # ── Inject PNG content type
    ct_xml = inject_content_type(ct_xml, "png")

    # ── Process each figure
    base_rel_id = find_max_rel_id(rels_xml) + 1
    base_draw_id = 100  # start image draw IDs at 100 to avoid clashes

    inserted_xml_parts = [section_heading_xml("§5.7  Evaluation Figures")]

    for fig_idx, (fname, caption) in enumerate(FIGURES):
        src_path    = os.path.join(FIGURES_DIR, fname)
        media_name  = f"hover_{fname}"          # unique name in word/media/
        rel_id_str  = f"rId{base_rel_id + fig_idx}"
        draw_id     = base_draw_id + fig_idx

        # Get pixel dimensions → EMU
        w_px, h_px = png_dimensions(src_path)
        cx_emu = min(px_to_emu(w_px), MAX_WIDTH_EMU)
        cy_emu = int(h_px * cx_emu / px_to_emu(w_px))   # keep aspect ratio

        print(f"  {fname}: {w_px}×{h_px} px → {cx_emu//914400:.1f}\" × {cy_emu//914400:.1f}\"")

        # Add image bytes
        with open(src_path, "rb") as fh:
            file_contents[f"word/media/{media_name}"] = fh.read()

        # Add relationship
        rels_xml = inject_rel(rels_xml, new_rel_entry(rel_id_str, media_name))

        # Build figure + caption XML
        if fig_idx > 0:
            # Small spacer between figures
            inserted_xml_parts.append(
                '<w:p><w:pPr><w:spacing w:before="240" w:after="0"/></w:pPr></w:p>'
            )
        inserted_xml_parts.append(
            image_paragraph_xml(rel_id_str, cx_emu, cy_emu, fname, draw_id)
        )
        inserted_xml_parts.append(caption_paragraph_xml(caption))

    # ── Inject figure block before </w:body>
    figures_block = "\n".join(inserted_xml_parts)
    doc_xml = doc_xml.replace("</w:body>", figures_block + "\n</w:body>")

    # ── Update modified files
    file_contents["word/document.xml"]              = doc_xml.encode("utf-8")
    file_contents["word/_rels/document.xml.rels"]   = rels_xml.encode("utf-8")
    file_contents["[Content_Types].xml"]            = ct_xml.encode("utf-8")

    # 4. Write new zip
    with zipfile.ZipFile(OUTPUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    print(f"\nDone.  v7 written to:\n  {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
