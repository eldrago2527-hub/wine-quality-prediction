"""
Builds the final PDF report (same layout as the sample: Introduction, Objectives,
Tools, Program Code, Output, Conclusion) from the REAL results created by
wine_quality.py.   Run order:  python wine_quality.py  ->  python build_report.py
Edit the details in the block below before running.
"""
import os
import pandas as pd  
from reportlab.lib.pagesizes import A4  
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Image, Table, TableStyle, Preformatted, KeepTogether)
from PIL import Image as PILImage

# ============ EDIT THESE ============
STUDENT_NAME   = "Rahul Krishna S"
REGISTER_NO    = "71052510016"
DEPARTMENT     = "Department of Computer Science and Engineering (AIML)"
COLLEGE        = "Coimbatore Institute of Engineering and Technology"
SUBJECT_NAME   = "Data Science"
COURSE_CODE    = "U23AMT01"
YEAR_SEM       = "II/III"
DATE           = "16-10-2026"
GITHUB_LINK    = "https://github.com/eldrago2527-hub/wine-quality-prediction"
# ====================================

OUT = "outputs"
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=12,
                      leading=18, alignment=TA_JUSTIFY, spaceAfter=6)
H = ParagraphStyle("h", parent=ss["Normal"], fontName="Times-Bold", fontSize=14, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=H, fontSize=12.5)
CEN = ParagraphStyle("c", parent=BODY, alignment=TA_CENTER)
TITLE = ParagraphStyle("t", parent=CEN, fontName="Times-Bold", fontSize=18, leading=24)
BUL = ParagraphStyle("b", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=8.6)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def fig(name, caption, width=13 * cm):
    path = os.path.join(OUT, name)
    w, h = PILImage.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w),
                         Paragraph(f"<i>{caption}</i>", CEN), Spacer(1, 6)])


def table(df):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b70")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f7ff")]),
    ]))
    return t


reg = pd.read_csv(f"{OUT}/regression_results.csv")
clf = pd.read_csv(f"{OUT}/classification_results.csv")
imp = pd.read_csv(f"{OUT}/feature_importance.csv")
summ = pd.read_csv(f"{OUT}/dataset_summary.csv").round(2)
summ = summ[["Unnamed: 0", "count", "mean", "std", "min", "max"]].rename(columns={"Unnamed: 0": "feature"})
br, bc = reg.iloc[0], clf.iloc[0]
top3 = ", ".join(imp["Unnamed: 0"].head(3)) if "Unnamed: 0" in imp else ", ".join(imp.iloc[:3, 0])
n_rows = int(summ.loc[summ.feature == "quality", "count"].iloc[0])

s = []
# ---------------- COVER ----------------
s += [Spacer(1, 1.5 * cm), Paragraph("ASSIGNMENT-2: MINI PROJECT", TITLE), Spacer(1, 12),
      Paragraph("WINE QUALITY PREDICTION", TITLE), Spacer(1, 1 * cm),
      Paragraph("Submitted by", CEN),
      Paragraph(f"<b>{STUDENT_NAME}</b>", CEN),
      Paragraph(f"Register Number: {REGISTER_NO}", CEN),
      Paragraph(DEPARTMENT, CEN), Paragraph(COLLEGE, CEN), Spacer(1, 1 * cm),
      Paragraph(f"Subject Name : {SUBJECT_NAME}", CEN),
      Paragraph(f"Course Code : {COURSE_CODE}", CEN),
      Paragraph(f"Year/Semester : {YEAR_SEM}", CEN),
      Paragraph(f"Date : {DATE}", CEN), Spacer(1, 1 * cm),
      Paragraph(f"<b>GitHub Repository:</b> <link href='{GITHUB_LINK}' color='blue'>{GITHUB_LINK}</link>", CEN),
      PageBreak()]

# ---------------- INTRODUCTION ----------------
s += [Paragraph("INTRODUCTION:", H),
      Paragraph("Wine Quality Prediction is a machine learning project that predicts the quality of wine "
                "from its physicochemical properties such as acidity, sugar, chlorides, sulfur dioxide, density, "
                "pH, sulphates and alcohol. The project uses the UCI Wine Quality dataset, which contains red and "
                "white variants of the Portuguese 'Vinho Verde' wine, with a quality score given by wine experts.", BODY),
      Paragraph("The problem is solved using two approaches. In the <b>regression</b> approach, the numeric quality "
                "score is predicted directly. In the <b>classification</b> approach, the wine is labelled as "
                "<b>Good</b> (quality of 6 or above) or <b>Not Good</b> (quality below 6). The performance of several models is "
                "compared using standard evaluation metrics.", BODY),
      Paragraph("OBJECTIVES:", H)]
s += bullets([
    "To load, clean and explore the UCI Wine Quality dataset.",
    "To visualise the quality distribution and the relationship between chemical properties and quality.",
    "To build regression models (Linear, Ridge, Random Forest, Gradient Boosting) to predict the quality score.",
    "To build classification models (Logistic Regression, KNN, Decision Tree, SVM, Random Forest) to predict Good/Not Good wine.",
    "To compare regression and classification approaches using MAE, RMSE, R-squared, Accuracy, Precision, Recall and F1-score.",
    "To identify the chemical properties that influence wine quality the most."])
s += [Paragraph("TOOLS AND TECHNOLOGIES:", H)]
s += bullets([
    "Programming Language: Python 3",
    "Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn",
    "Dataset: UCI Wine Quality Dataset (winequality-red.csv and winequality-white.csv)",
    "Development Environment: VS Code / Jupyter Notebook",
    "Version Control: Git and GitHub",
    "Platform: Windows Operating System"])
s += [Paragraph("DATASET DESCRIPTION:", H),
      Paragraph(f"After combining the red and white wine files and removing duplicate records, the dataset contains "
                f"<b>{n_rows}</b> samples with 11 input features (fixed acidity, volatile acidity, citric acid, residual "
                f"sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol), an added "
                f"<i>wine_type</i> feature (0 = red, 1 = white) and the target <i>quality</i>.", BODY),
      table(summ), PageBreak()]

# ---------------- CODE ----------------
s += [Paragraph("PROGRAM CODE:", H), Paragraph("wine_quality.py", H2)]
s += [Preformatted(open("wine_quality.py", encoding="utf-8").read(), CODE), PageBreak()]

# ---------------- OUTPUT ----------------
s += [Paragraph("OUTPUT:", H),
      Paragraph("Quality Distribution", H2),
      fig("01_quality_distribution.png", "Figure 1: Distribution of wine quality scores"),
      Paragraph("The bar chart shows how many wines belong to each quality score. Most wines receive "
                "average scores, while very low and very high scores are rare, so the data is imbalanced.", BODY),
      Paragraph("Correlation Heatmap", H2),
      fig("02_correlation_heatmap.png", "Figure 2: Correlation between features and quality", 14 * cm),
      Paragraph("The heatmap shows the linear relationship between every pair of features. It helps to find the "
                "properties that are most related to wine quality.", BODY),
      Paragraph("Alcohol vs Quality", H2),
      fig("03_alcohol_vs_quality.png", "Figure 3: Alcohol content for each quality level", 11 * cm),
      PageBreak(),
      Paragraph("Regression Results", H2), table(reg), Spacer(1, 8),
      Paragraph(f"The regression models predict the quality score as a number. <b>{br['Model']}</b> gave the best "
                f"performance with R<super>2</super> = {br['R2']}, MAE = {br['MAE']} and RMSE = {br['RMSE']}. When the predicted "
                f"value is rounded to the nearest whole score, the exact-score accuracy is {br['Rounded Accuracy']}.", BODY),
      fig("04_actual_vs_predicted.png", "Figure 4: Actual vs predicted quality (best regression model)", 9 * cm),
      Paragraph("Classification Results", H2), table(clf), Spacer(1, 8),
      Paragraph(f"The classification models label each wine as Good or Not Good. <b>{bc['Model']}</b> achieved the "
                f"highest test accuracy of {bc['Accuracy']} with precision {bc['Precision']}, recall {bc['Recall']} "
                f"and F1-score {bc['F1-Score']}. The 5-fold cross-validation accuracy is also reported to confirm "
                f"that the result is stable.", BODY),
      fig("05_confusion_matrix.png", "Figure 5: Confusion matrix (best classification model)", 8.5 * cm),
      Paragraph("Feature Importance", H2),
      fig("06_feature_importance.png", "Figure 6: Feature importance from Random Forest", 11 * cm),
      Paragraph(f"The most important features for predicting wine quality are: <b>{top3}</b>.", BODY),
      Paragraph("Regression vs Classification", H2),
      Paragraph("Regression keeps the full 0–10 score and shows how far the prediction is from the real score, but "
                "predicting the exact score is difficult because the classes are imbalanced and neighbouring "
                "scores are very similar. Classification into Good / Not Good is simpler and gives a clearer "
                "decision, and is more useful when only a pass/fail decision on quality is required.", BODY),
      Paragraph("CONCLUSION:", H),
      Paragraph(f"The Wine Quality Prediction project was successfully developed using Python and Scikit-learn. "
                f"Both regression and classification approaches were applied to the UCI Wine Quality dataset and "
                f"compared. {br['Model']} was the best regression model (R<super>2</super> = {br['R2']}) and {bc['Model']} was the "
                f"best classifier (accuracy = {bc['Accuracy']}). Alcohol and other chemical properties were found "
                f"to strongly influence quality. The project shows that machine learning can support wine quality "
                f"assessment using measurable chemical properties.", BODY),
      Paragraph(f"<b>GitHub Link:</b> <link href='{GITHUB_LINK}' color='blue'>{GITHUB_LINK}</link>", BODY)]


def footer(c, d):
    c.setFont("Times-Roman", 9)
    c.drawCentredString(A4[0] / 2, 1 * cm, f"Page {d.page}")


SimpleDocTemplate("Wine_Quality_Prediction_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(s, onFirstPage=footer, onLaterPages=footer)
print("Report created: Wine_Quality_Prediction_Report.pdf")
