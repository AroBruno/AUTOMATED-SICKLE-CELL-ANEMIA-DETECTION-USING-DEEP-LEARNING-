import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load Model Once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\hi nana project\Sickle-Cell-anemia-detection-using-Efficietnet-based-CNN-architecture\sickle_cell_detection_model.keras")

model = load_model()


# Create Folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Login", "Upload Image", "Patient Details & Report"])

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

# Function to Save Uploaded File
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Prediction Function
def predict_sickle_cell(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return round(prediction * 100, 2), round(100 - prediction * 100, 2)

# Generate PDF Report


def generate_pdf(patient, sickle, normal, doctor, hospital):
    pdf_filename = f"report_{patient['id']}.pdf"
    pdf_path = os.path.join("reports", pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph(f"{hospital} - Medical Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    data = [
        ["Hospital Name:", hospital],
        ["Doctor Name:", doctor],
        ["Patient Name:", patient['name']],
        ["Patient ID:", patient['id']],
        ["Age:", patient['age']],
        ["Sex:", patient['sex']],
        ["Additional Details:", patient['details']],
        ["Sickle Cell Probability:", f"{sickle}%"],
        ["Normal Cell Probability:", f"{normal}%"],
        ["Advice:", "Consult a specialist if necessary."]
    ]
    
    table = Table(data, colWidths=[150, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Thank You Note
    thank_you_note = Paragraph("Thank you for choosing our hospital. Wishing you good health!", normal_style)
    elements.append(thank_you_note)
    
    doc.build(elements)
    return pdf_path


# Login Page
if page == "Login":
    st.title("🔑 Doctor Login")
    username = st.text_input("Username", value="hematopathologists")
    doctor = st.text_input("Doctor Name")
    hospital = st.text_input("Hospital Name")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "hematopathologists" and password == "pass123":
            st.session_state.logged_in = True
            st.session_state.doctor = doctor
            st.session_state.hospital = hospital
            st.success("Login Successful! Use sidebar to navigate.")
        else:
            st.error("Invalid Credentials!")

# Upload & Prediction Page
elif page == "Upload Image":
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login first!")
    else:
        st.title("🩸 Sickle Cell Detection")
        uploaded_file = st.file_uploader("Upload a Blood Cell Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            st.image(file_path, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Image"):
                with st.spinner("Processing Image..."):
                    sickle, normal = predict_sickle_cell(file_path)
                    st.session_state.sickle = sickle
                    st.session_state.normal = normal
                    st.session_state.image_uploaded = True
                    st.success(f"Sickle Cell Probability: {sickle}%")
                    st.success(f"Normal Cell Probability: {normal}%")
                    if sickle > 50:
                        st.error("⚠️ High Risk of Sickle Cell Anemia Detected!")
                    st.sidebar.success("Proceed to 'Patient Details & Report'")

# Patient Details & Report Page
elif page == "Patient Details & Report":
    if not st.session_state.logged_in:
        st.warning("⚠️ Please login first!")
    elif not st.session_state.image_uploaded:
        st.warning("⚠️ Please upload and analyze an image first!")
    else:
        st.title("📝 Patient Details & Report Generation")
        patient = {
            "name": st.text_input("Patient Name"),
            "id": st.text_input("Patient ID"),
            "age": st.slider('📅 Age', 5, 80),
            "sex": st.selectbox("Sex", ["Male", "Female", "Other"]),
            "details": st.text_area("Additional Information")
        }
        if st.button("Generate Report"):
            if all(patient.values()):
                pdf_path = generate_pdf(
                    patient, 
                    st.session_state.sickle, 
                    st.session_state.normal, 
                    st.session_state.get("doctor", "Unknown"), 
                    st.session_state.get("hospital", "Unknown")
                )
                st.success("Report Generated Successfully!")
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("Download Report", pdf_file, file_name=f"{patient['id']}_report.pdf", mime="application/pdf")
            else:
                st.error("Please fill all fields!")
