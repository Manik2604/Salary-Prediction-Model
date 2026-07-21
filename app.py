import os
import joblib
import pandas as pd
import gradio as gr

# ===========================
# Load Model
# ===========================

MODEL_PATH = "Salary_Prediction_Model.pkl"

model = joblib.load(MODEL_PATH)

# ===========================
# Prediction
# ===========================

def predict_salary(age, experience, education):

    data = {
        "Age": float(age),
        "Years of Experience": float(experience),

        "Job Title_Other": 1,

        "Education Level_Bachelor's": 0,
        "Education Level_Bachelor's Degree": 0,
        "Education Level_High School": 0,
        "Education Level_Master's": 0,
        "Education Level_Master's Degree": 0,
        "Education Level_Other": 0,
        "Education Level_PhD": 0,
        "Education Level_phD": 0,
    }

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
        data["Education Level_Other"] = 1

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return f"₹ {prediction:,.2f}"


# ===========================
# CSS
# ===========================

css = """
.gradio-container{
background-image:url("https://images.unsplash.com/photo-1521791136064-7986c2920216?q=80&w=2070&auto=format&fit=crop");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

.box{
background:rgba(255,255,255,.93);
padding:25px;
border-radius:18px;
}

footer{visibility:hidden;}
"""

# ===========================
# UI
# ===========================

with gr.Blocks(css=css,title="Salary Prediction") as demo:

    with gr.Column(elem_classes="box"):

        gr.Markdown("# 💼 Salary Prediction")

        age = gr.Number(label="Age",value=25)

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
                "Other"
            ],
            value="Bachelor's",
            label="Education"
        )

        btn = gr.Button("Predict Salary")

        output = gr.Textbox(label="Prediction")

        btn.click(
            predict_salary,
            [age,experience,education],
            output
        )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT",7860))
)
