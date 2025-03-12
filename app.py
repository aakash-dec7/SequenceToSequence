from src.seq2seq.logger import logger
from flask import Flask, render_template, request
from src.seq2seq.prediction.prediction import Prediction
from src.seq2seq.config.configuration import ConfigurationManager


app = Flask(__name__)

# Load Prediction Model ONCE
prediction_model = Prediction(config=ConfigurationManager().get_prediction_config())


@app.route("/", methods=["GET"])
def home_page():
    """Render the main translation page."""
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    """Handle translation request."""
    try:
        english_text = request.form.get("english_text", "").strip()
        if not english_text:
            return render_template(
                "index.html", translated_text="Error: No input provided."
            )

        # Perform translation
        translated_sentence = prediction_model.predict(english_text)

        return render_template("index.html", translated_text=translated_sentence)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template(
            "index.html", translated_text="An error occurred. Please try again later."
        )


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=3000, debug=True)
    app.run(host="0.0.0.0", port=8080)
