import os
import joblib
import pandas as pd
import gradio as gr

# ============================================
# Load Model
# ============================================

MODEL_PATH = "Salary_Prediction_Model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ============================================
# Prediction Function
# ============================================

def predict_salary(age, experience, education, job):

    try:
        age = float(age)
        experience = float(experience)

        data = {
            "Age": age,
            "Years of Experience": experience,

            "Education Level_Bachelor's": 0,
            "Education Level_Bachelor's Degree": 0,
            "Education Level_High School": 0,
            "Education Level_Master's": 0,
            "Education Level_Master's Degree": 0,
            "Education Level_PhD": 0,
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

        return f"💰 Predicted Salary: ₹ {prediction:,.2f}"

    except Exception as e:
        return f"Error: {e}"


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

.box{
    background:rgba(255,255,255,0.95);
    padding:20px;
    border-radius:18px;
    box-shadow:0 0 20px rgba(0,0,0,.3);
}

.gradio-container *{
    color:black !important;
}

label{
    font-weight:bold !important;
}

textarea,
input,
select{
    background:white !important;
    color:black !important;
}

button{
    font-weight:bold !important;
}

footer{
    visibility:hidden;
}
"""

# ============================================
# Interface
# ============================================

with gr.Blocks(
    title="Salary Prediction",
    css=css
) as demo:

    with gr.Column(elem_classes="box"):

        gr.Markdown(
            """
# 💼 Employee Salary Prediction

Predict employee salary using a Machine Learning model.
"""
        )

        with gr.Row():

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
                    choices=[
                        "Bachelor's",
                        "Bachelor's Degree",
                        "High School",
                        "Master's",
                        "Master's Degree",
                        "PhD",
                        "others",
                    ],
                    value="Bachelor's",
                    label="Education Level",
                )

                job = gr.Dropdown(
                    choices=[
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
                        "others",
                    ],
                    value="Software Engineer",
                    label="Job Title",
                )

                btn = gr.Button(
                    "Predict Salary",
                    variant="primary"
                )

                output = gr.Textbox(
                    label="Prediction",
                    lines=2
                )

            with gr.Column(scale=1):

                gr.Markdown(
                    """
# 👨‍💻 Developer Details

**Name:** Manik Jindal

**College:**  
Panipat Institute of Engineering and Technology

---

### Project

Salary Prediction using Linear Regression

---

### Technologies

- Python
- Pandas
- Scikit-Learn
- Joblib
- Gradio

---

### Input

- Age
- Experience
- Education
- Job Title

---

### Output

Predicted Employee Salary
"""
                )

        btn.click(
            fn=predict_salary,
            inputs=[
                age,
                experience,
                education,
                job
            ],
            outputs=output
        )

# ============================================
# Launch
# ============================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )
