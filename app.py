import os
import joblib
import pandas as pd
import gradio as gr


# =====================================
# Load Model
# =====================================

MODEL_PATH = "Salary_Prediction_Model.pkl"

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    raise Exception(
        f"Model file loading error: {e}\n"
        "Keep Salary_Prediction_Model.pkl in the same folder."
    )



# =====================================
# Prediction Function
# =====================================

def predict_salary(age, experience, education):

    try:

        # Raw input dataframe
        input_data = pd.DataFrame(
            {
                "Age": [float(age)],
                "Years of Experience": [float(experience)],
                "Education Level": [education]
            }
        )


        # Prediction
        prediction = model.predict(input_data)[0]


        return f"💰 Predicted Salary: ₹ {prediction:,.2f}"


    except Exception as e:

        return f"❌ Error: {str(e)}"



# =====================================
# CSS
# =====================================

css = """

.gradio-container{

    background:#eaf2f8;

}


.container{

    background:white;

    padding:30px;

    border-radius:20px;

    box-shadow:0px 8px 20px rgba(0,0,0,0.15);

}


/* All text black */

.container *{

    color:black !important;

}


footer{

    display:none;

}

"""



# =====================================
# Gradio Interface
# =====================================

with gr.Blocks(
    css=css,
    title="Salary Prediction"
) as demo:


    with gr.Column(
        elem_classes="container"
    ):


        gr.Markdown(
            """
            # 💼 Salary Prediction System

            Predict employee salary using Machine Learning.
            """
        )


        gr.Markdown("---")


        with gr.Row():


            # Input side

            with gr.Column():


                gr.Markdown(
                    "## 📥 Employee Details"
                )


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
                        "Other"
                    ],

                    value="Bachelor's",

                    label="Education Level"

                )


                button = gr.Button(
                    "🚀 Predict Salary"
                )


                output = gr.Textbox(
                    label="Prediction Result"
                )



            # Developer details

            with gr.Column():


                gr.Markdown(
                    """
                    ## 👨‍💻 Developer Details


                    **Name:**  
                    Manik Jindal


                    **College:**  
                    Panipat Institute of Engineering and Technology


                    **Project:**  
                    Salary Prediction using Machine Learning


                    ## 🛠 Technology Stack


                    - Python
                    - Pandas
                    - Scikit-Learn
                    - Machine Learning
                    - Joblib
                    - Gradio


                    ## 📌 About Project


                    This application predicts salary using:

                    👤 Age

                    💼 Years of Experience

                    🎓 Education Level


                    The trained Machine Learning model
                    automatically processes the input data.
                    """
                )



        button.click(

            fn=predict_salary,

            inputs=[
                age,
                experience,
                education
            ],

            outputs=output

        )



# =====================================
# Launch
# =====================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )

    )
