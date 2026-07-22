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
    try:
        input_data = pd.DataFrame({
            "Age": [age],
            "Years of Experience": [experience],
            "Education Level": [education],
            "Job Title": [job]
        })

        prediction = model.predict(input_data)[0]

        return f"💰 Predicted Salary: ₹ {prediction:,.2f}"

    except Exception as e:
        return f"❌ Error:\n{e}"


# ============================================
# CSS
# ============================================

css = """
.gradio-container{
    background:#f2f2f2;
}

.box{
    background:white;
    padding:20px;
    border-radius:15px;
}
"""


# ============================================
# Interface
# ============================================

with gr.Blocks(css=css, title="Salary Prediction") as demo:

    with gr.Column(elem_classes="box"):

        gr.Markdown("# 💼 Employee Salary Prediction")

        with gr.Row():

            with gr.Column():

                age = gr.Number(label="Age", value=25)

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
                        "PhD"
                    ],
                    value="Bachelor's",
                    label="Education Level"
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
                        "Software Engineer Manager"
                    ],
                    value="Software Engineer",
                    label="Job Title"
                )

                btn = gr.Button("Predict Salary")

                output = gr.Textbox(label="Prediction")

            with gr.Column():

                gr.Markdown("""
## 👨‍💻 Developer

**Name:** Manik

**College:** Panipat Institute of Engineering and Technology

### Project

Employee Salary Prediction using Linear Regression

### Technology

- Python
- Pandas
- Scikit-Learn
- Gradio
- Joblib
""")

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
        server_port=int(os.environ.get("PORT", 7860))
    )
