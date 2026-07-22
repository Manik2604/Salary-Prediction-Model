import os
import gradio as gr
import joblib
import pandas as pd

# ============================================
# Load Trained Model
# ============================================
model = joblib.load("Salary_Prediction_Model.pkl")


# ============================================
# Prediction Function
# ============================================
def predict_salary(age, experience, education, job):

    data = {
        "Age": age,
        "Years of Experience": experience,

        "Education Level_Bachelor's": 0,
        "Education Level_Bachelor's Degree": 0,
        "Education Level_High School": 0,
        "Education Level_Master's": 0,
        "Education Level_Master's Degree": 0,
        "Education Level_PhD": 0,
        "Education Level_phD": 0,
        "Education Level_others": 0,

        "Job Title_Back end Developer": 0,
        "Job Title_Data Analyst": 0,
        "Job Title_Data Scientist": 0,
        "Job Title_Full Stack Engineer": 0,
        "Job Title_Marketing Manager": 0,
        "Job Title_Product Manager": 0,
        "Job Title_Senior Project Engineer": 0,
        "Job Title_Senior Software Engineer": 0,
        "Job Title_Software Engineer": 0,
        "Job Title_Software Engineer Manager": 0,
        "Job Title_others": 0,
    }

    # Education Encoding
    if education == "Bachelor's":
        data["Education Level_Bachelor's"] = 1

    elif education == "Bachelor's Degree":
        data["Education Level_Bachelor's Degree"] = 1

    elif education == "High School":
        data["Education Level_High School"] = 1

    elif education == "Master's":
        data["Education Level_Master's"] = 1

    elif education == "Master's Degree":
        data["Education Level_Master's Degree"] = 1

    elif education == "PhD":
        data["Education Level_PhD"] = 1

    else:
        data["Education Level_others"] = 1

    # Job Encoding
    job_column = f"Job Title_{job}"

    if job_column in data:
        data[job_column] = 1
    else:
        data["Job Title_others"] = 1

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return f"💰 Predicted Salary : ₹ {prediction:,.2f}"


# ============================================
# CSS
# ============================================
css = """
.gradio-container{
    background-image:url("https://images.unsplash.com/photo-1521791136064-7986c2920216?q=80&w=2070&auto=format&fit=crop");
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

/* Main Box */
.box{
    background:rgba(255,255,255,0.95);
    padding:20px;
    border-radius:18px;
    box-shadow:0px 0px 20px rgba(0,0,0,0.3);
}

/* Make ALL text BLACK */
.gradio-container,
.gradio-container *{
    color:yellow !important;
}

/* Markdown */
.prose,
.prose p,
.prose h1,
.prose h2,
.prose h3,
.prose h4,
.prose strong,
.prose li{
    color:black !important;
}

/* Labels */
label{
    color:black !important;
    font-weight:bold !important;
}

/* Textbox */
textarea,
input{
    color:black !important;
    background:white !important;
}

/* Dropdown */
select{
    color:black !important;
    background:white !important;
}

/* Output Textbox */
textarea{
    font-weight:bold;
}

/* Button */
button{
    color:black !important;
    font-weight:bold !important;
}

/* Hide Footer */
footer{
    visibility:hidden;
}
"""


# ============================================
# Interface
# ============================================

with gr.Blocks(
    css=css,
    title="Salary Prediction"
) as demo:

    with gr.Column(elem_classes="box"):

        gr.Markdown(
            """
# 💼 Employee Salary Prediction

Predict the salary of an employee using a Machine Learning model.
"""
        )

        with gr.Row():

            # Left Side
            with gr.Column(scale=2):

                age = gr.Number(
                    label="Age",
                    value=25
                )

                experience = gr.Number(
                    label="Years of Experience",
                    value=2
                )

                education = gr.Dropdown(
                    [
                        "Bachelor's",
                        "Bachelor's Degree",
                        "High School",
                        "Master's",
                        "Master's Degree",
                        "PhD",
                        "Others",
                    ],
                    label="Education Level",
                    value="Bachelor's",
                )

                job = gr.Dropdown(
                    [
                        "Back end Developer",
                        "Data Analyst",
                        "Data Scientist",
                        "Full Stack Engineer",
                        "Marketing Manager",
                        "Product Manager",
                        "Senior Project Engineer",
                        "Senior Software Engineer",
                        "Software Engineer",
                        "Software Engineer Manager",
                        "Others",
                    ],
                    label="Job Title",
                    value="Software Engineer",
                )

                btn = gr.Button(
                    "Predict Salary",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Prediction",
                    lines=2
                )

            # Right Side
            with gr.Column(scale=1):

                gr.Markdown(
                    """
# 👩‍💻 Developer Details

*Name:* Manik

*College:*  
Panipat Institute of Engineering and Technology

---

## 📌 Project

Salary Prediction using Linear Regression

---

## 🛠️ Technology Used

- Python
- Pandas
- Scikit-Learn
- Joblib
- Gradio

---

## 📥 Input Features

- Age
- Years of Experience
- Education Level
- Job Title

---

## 📤 Output

Predicted Employee Salary
"""
                )

        btn.click(
            fn=predict_salary,
            inputs=[
                age,
                experience,
                education,
                job,
            ],
            outputs=output,
        )


# ============================================
# Launch
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
