from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

def generate_pdf(summary_text, chart_paths, output_path="analysis_report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph("<b>Research Assistant Analysis Report</b>", styles["Title"]))
    content.append(Spacer(1, 0.3 * inch))

    # Summary
    content.append(Paragraph("<b>Summary:</b>", styles["Heading2"]))
    content.append(Paragraph(summary_text, styles["Normal"]))
    content.append(Spacer(1, 0.2 * inch))

    # Charts
    if chart_paths:
        content.append(Paragraph("<b>Charts:</b>", styles["Heading2"]))
        for chart in chart_paths:
            if os.path.exists(chart):
                content.append(Image(chart, width=5*inch, height=3*inch))
                content.append(Spacer(1, 0.2 * inch))

    doc.build(content)
    return output_path
