from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import datetime

def generate_pdf_report(metrics, logs, figures=None):
    """
    Generates a PDF summary of the experiment.
    metrics: dict of KPIs (Accuracy, TPS, etc.)
    logs: list of string logs
    figures: dict of {name: bytes_io_image}
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- HEADER ---
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "DB-BOA Thesis Experiment Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, height - 80, width - 50, height - 80)

    # --- KPI METRICS ---
    y_pos = height - 120
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "1. System Performance Metrics")
    y_pos -= 25

    c.setFont("Helvetica", 12)
    for key, value in metrics.items():
        c.drawString(70, y_pos, f"{key}: {value}")
        y_pos -= 20

    # --- PLOTS (If available) ---
    if figures:
        y_pos -= 20
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "2. Visual Validation")
        y_pos -= 150  # Space for image

        # Draw first plot found (usually ROC or Latency)
        # (In a real app, we'd loop, but we fit one for the demo)
        for name, img_bytes in figures.items():
            try:
                img = ImageReader(img_bytes)
                c.drawImage(img, 50, y_pos, width=300, height=150)
                c.drawString(50, y_pos - 15, f"Figure: {name}")
            except:
                c.drawString(50, y_pos, "[Image Error]")
            break 

    # --- LOGS ---
    y_pos -= 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "3. Transaction Execution Logs")
    y_pos -= 25

    c.setFont("Courier", 9)
    for log in logs[-15:]: # Last 15 logs to fit page
        c.drawString(50, y_pos, log)
        y_pos -= 12
        if y_pos < 50: # New page if full
            c.showPage()
            y_pos = height - 50

    c.save()
    buffer.seek(0)
    return buffer
