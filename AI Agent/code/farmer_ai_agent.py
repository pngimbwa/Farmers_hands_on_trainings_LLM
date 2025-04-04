import os
import json
import requests
from typing import Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pickle
import gradio as gr

# Clear any existing environment variable to isolate the .env file
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variable
load_dotenv()

# OPENAI API key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------------------------------------------
# Define response formats using Pydantic models
# --------------------------------------------------------------
class Weather(BaseModel):
    temperature: float = Field(description="Current temperature in Celcius")
    wind_speed: float = Field(description="Current wind speed in m/s")


class SoilData(BaseModel):
    moisture_level: float = Field(description="Soil moisture percentage (0-100)")
    temperature: float = Field(description="Soil temperature in Celcius")


class IrrigationResponse(BaseModel):
    temperature: float = Field(description="Current air temperature in Celsius")
    soil_moisture: float = Field(description="Current soil moisture percentage")
    irrigation_needed: bool = Field(description="Whether irrigation is recommended")
    water_amount: float = Field(
        description="Recommended water amount in liters per square meter"
    )
    responses: str = Field(
        description="Natural language response with irrigation advice"
    )


# --------------------------------------------------------------
# Define tools (functions)
# --------------------------------------------------------------
def get_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather data for given coordinates."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}¤t=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return {
        "temperature": data["current"]["temperature_2m"],
        "wind_speed": data["current"]["wind_speed_10m"],
    }


def get_soil_data(sensor_id: str) -> dict:
    """Simulated soil sensor data (replace with real sensor API/integration)"""
    import random

    return {
        "moisture_level": random.uniform(20, 80),
        "temperature": random.uniform(15, 30),
    }


def control_irrigation(water_amount: float) -> dict:
    """Simulated irrigation control"""
    return {"status": "success", "water_amount": water_amount}


def call_function(name: str, args: dict) -> dict:
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_soil_data":
        return get_soil_data(**args)
    elif name == "control_irrigation":
        return control_irrigation(**args)
    return {}


# --------------------------------------------------------------
# Define tools for the AI agent
# --------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_soil_data",
            "description": "Get current soil sensor readings",
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor_id": {"type": "string"},
                },
                "required": ["sensor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_irrigation",
            "description": "Activate irrigation system with specified water amount",
            "parameters": {
                "type": "object",
                "properties": {
                    "water_amount": {"type": "number"},
                },
                "required": ["water_amount"],
            },
        },
    },
]


# --------------------------------------------------------------
# Learning mechanism
# --------------------------------------------------------------
class LearningAgent:
    def __init__(self):
        self.data_file = "irrigation_learning_data.pkl"
        self.default_rules = {
            "cotton": {
                "min_moisture": 50,
                "base_water": 6.0,
                "temp_threshold": 28,
                "wind_adjust": 0.8,
            },
            "corn": {
                "min_moisture": 40,
                "base_water": 5.0,
                "temp_threshold": 25,
                "wind_adjust": 0.9,
            },
            "soybeans": {
                "min_moisture": 45,
                "base_water": 5.5,
                "temp_threshold": 27,
                "wind_adjust": 0.85,
            },
        }
        self.learned_rules = self.load_learned_data()

    def load_learned_data(self) -> Dict:
        """Load previously saved rules or return defaults if file doesn't exist."""
        try:
            with open(self.data_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return self.default_rules.copy()

    def save_learned_data(self):
        """Save current rules to file."""
        with open(self.data_file, "wb") as f:
            pickle.dump(self.learned_rules, f)

    def update_rules(
        self,
        crop: str,
        moisture: float,
        temp: float,
        water_used: float,
        feedback: float,
    ):
        """Update rules based on feedback."""
        rules = self.learned_rules[crop]
        if feedback < 0.7:  # Adjust if outcome was suboptimal
            if moisture < rules["min_moisture"]:  # Too dry
                rules["min_moisture"] -= 1.0
                rules["base_water"] += 0.2
            elif moisture > rules["min_moisture"] + 10:  # Too wet
                rules["min_moisture"] += 1.0
                rules["base_water"] -= 0.2
            if temp > rules["temp_threshold"] and feedback < 0.6:  # Too hot
                rules["temp_threshold"] -= 0.5
        self.save_learned_data()


# --------------------------------------------------------------
# System prompt
# --------------------------------------------------------------
system_prompt = """
You are an adaptive irrigation AI agent. Use weather and soil data to:
1. Assess conditions for a specific crop (cotton, corn, soybeans)
2. Adapt irrigation decisions based on learned rules, which may evolve:
   - Start with: Cotton (50% moisture, 6L/m²), Corn (40%, 5L/m²), Soybeans (45%, 5.5L/m²)
   - Adjust water: +15-20% if temp exceeds crop threshold, -10-20% if wind > 5 m/s
3. Learn from feedback to refine thresholds and water amounts
4. Provide detailed, crop-specific recommendations
"""


# --------------------------------------------------------------
# Core irrigation logic
# --------------------------------------------------------------
def manage_irrigation(
    location: str,
    crop: str,
    agent: LearningAgent,
    sensor_id: str = "sensor_001",
    latitude: float = 31.4505,  # Default: Tifton, GA
    longitude: float = -83.5085,
    feedback: Optional[float] = None,
):
    if crop.lower() not in agent.learned_rules:
        return IrrigationResponse(
            temperature=0,
            soil_moisture=0,
            irrigation_needed=False,
            water_amount=0,
            response=f"Error: {crop} not recognized. Supported crops: cotton, corn, soybeans",
        )

    # Update rules if feedback is provided
    if feedback is not None:
        last_data = agent.load_learned_data()
        agent.update_rules(
            crop,
            last_data.get("last_moisture", 50),
            last_data.get("last_temp", 25),
            last_data.get("last_water", 5),
            feedback,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Should I irrigate my {crop} field at {location} today? Current rules: {agent.learned_rules[crop]}",
        },
    ]

    # Initial call to gather data
    completion = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools
    )
    for tool_call in completion.choices[0].message.tool_calls or []:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if name == "get_weather":
            args = {"latitude": latitude, "longitude": longitude}
        elif name == "get_soil_data":
            args = {"sensor_id": sensor_id}
        result = call_function(name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    # Final response with decision
    completion_final = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        response_format=IrrigationResponse,
    )
    response = completion_final.choices[0].message.parsed

    # Store latest conditions for future feedback
    agent.learned_rules["last_moisture"] = response.soil_moisture
    agent.learned_rules["last_temp"] = response.temperature
    agent.learned_rules["last_water"] = response.water_amount
    agent.save_learned_data()

    return response


# --------------------------------------------------------------
# Gradio Interface
# --------------------------------------------------------------
agent = LearningAgent()


def check_irrigation(crop, location, latitude, longitude):
    response = manage_irrigation(
        location, crop, agent, latitude=latitude, longitude=longitude
    )
    output = (
        f"Temperature: {response.temperature}°C\n"
        f"Soil Moisture: {response.soil_moisture}%\n"
        f"Irrigation Needed: {'Yes' if response.irrigation_needed else 'No'}\n"
        f"Water Amount: {response.water_amount} L/m²\n"
        f"Recommendation: {response.response}"
    )
    return output


def provide_feedback(crop, feedback_score):
    if not 0 <= feedback_score <= 1:
        return "Error: Feedback score must be between 0 and 1"
    response = manage_irrigation(
        "Tifton", crop, agent, feedback=feedback_score
    )  # Default location for feedback
    output = (
        f"Feedback ({feedback_score}) applied for {crop}\n"
        f"Updated Recommendation: {response.response}\n"
        f"Updated Rules: {agent.learned_rules[crop]}"
    )
    return output


def view_rules(crop):
    if crop in agent.learned_rules:
        return f"Current rules for {crop}: {agent.learned_rules[crop]}"
    return "Error: Crop not recognized"


with gr.Blocks(title="Irrigation Management Tool") as demo:
    gr.Markdown("# Irrigation Management Tool")
    gr.Markdown(
        "This is an AI-powered tool to manage irrigation, not a chatbot. Input your data below to get recommendations."
    )

    # Tab 1: Check Irrigation
    with gr.Tab("Check Irrigation"):
        gr.Markdown("Enter details to see if your field needs irrigation.")
        crop_input = gr.Dropdown(
            choices=["cotton", "corn", "soybeans"], label="Crop Type"
        )
        location_input = gr.Textbox(label="Location", placeholder="e.g., Tifton")
        lat_input = gr.Number(label="Latitude", value=31.4505)
        lon_input = gr.Number(label="Longitude", value=-83.5085)
        check_button = gr.Button("Get Recommendation")
        output_text = gr.Textbox(label="Irrigation Recommendation", lines=5)
        check_button.click(
            fn=check_irrigation,
            inputs=[crop_input, location_input, lat_input, lon_input],
            outputs=output_text,
        )

    # Tab 2: Provide Feedback
    with gr.Tab("Provide Feedback"):
        gr.Markdown("After irrigating, provide feedback (0 to 1) to help the AI learn.")
        feedback_crop = gr.Dropdown(
            choices=["cotton", "corn", "soybeans"], label="Crop Type"
        )
        feedback_score = gr.Slider(
            minimum=0, maximum=1, step=0.1, label="Feedback Score (0 = bad, 1 = good)"
        )
        feedback_button = gr.Button("Submit Feedback")
        feedback_output = gr.Textbox(label="Feedback Result", lines=3)
        feedback_button.click(
            fn=provide_feedback,
            inputs=[feedback_crop, feedback_score],
            outputs=feedback_output,
        )

    # Tab 3: View Rules
    with gr.Tab("View Learned Rules"):
        gr.Markdown("See the current learned rules for each crop.")
        rules_crop = gr.Dropdown(
            choices=["cotton", "corn", "soybeans"], label="Crop Type"
        )
        rules_button = gr.Button("Show Rules")
        rules_output = gr.Textbox(label="Learned Rules", lines=2)
        rules_button.click(fn=view_rules, inputs=rules_crop, outputs=rules_output)

if __name__ == "__main__":
    demo.launch()
